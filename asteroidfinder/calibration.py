from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import csv

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy.ndimage import maximum_filter, median_filter, minimum_filter

from .io import AstroImage, load_image, save_fits


@dataclass(frozen=True)
class CalibrationResult:
    image: AstroImage
    data: np.ndarray
    hot_pixel_mask: np.ndarray


@dataclass(frozen=True)
class HotPixelFrameDiagnostic:
    path: Path
    candidate_count: int
    persistent_count: int
    transient_count: int
    mask_path: Path | None = None


@dataclass(frozen=True)
class HotPixelDiagnostic:
    shape: tuple[int, int]
    min_frames: int
    persistent_count: int
    transient_count: int
    persistent_mask_path: Path | None
    transient_mask_path: Path | None
    frame_diagnostics: list[HotPixelFrameDiagnostic]


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
    progress_callback: Callable[[int, int, str], None] | None = None,
    **kwargs: object,
) -> list[CalibrationResult]:
    """Calibrate images using one conservative persistent hot-pixel map.

    This is safer for star fields than single-frame hot-pixel detection: a pixel
    must be both isolated and repeatedly hot at the same sensor coordinate.
    """

    all_paths = list(paths)
    total = max(1, len(all_paths))
    done = 0
    grouped_paths = _group_paths_by_shape(all_paths)
    results = []
    if output_dir is not None:
        summary_path = Path(output_dir) / "hot_pixel_qa" / "hot_pixel_summary.csv"
        if summary_path.exists():
            summary_path.unlink()
    for shape_paths in grouped_paths.values():
        mask, diagnostic = _build_persistent_hot_pixel_mask_with_diagnostic(
            shape_paths,
            sigma=hot_sigma,
            neighbor_sigma=neighbor_sigma,
            min_center_neighbor_ratio=min_center_neighbor_ratio,
            min_frames=min_frames,
            diagnostic_dir=None if output_dir is None else Path(output_dir) / "hot_pixel_qa",
            progress_callback=(
                None
                if progress_callback is None
                else lambda step, steps, text: progress_callback(min(done + step, total), total, text)
            ),
        )
        done += len(shape_paths)
        results.extend(calibrate_image(path, hot_pixel_mask=mask, hot_sigma=hot_sigma, **kwargs) for path in shape_paths)
        if progress_callback is not None:
            progress_callback(min(done, total), total, "Applied hot-pixel mask")
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(total, total, "Writing calibrated FITS files")
        for result in results:
            save_fits(result.data, out_dir / f"{result.image.path.stem}_calibrated.fits", result.image.header)
    elif progress_callback is not None:
        progress_callback(total, total, "Hot-pixel cleaning complete")
    return results


def build_persistent_hot_pixel_mask(
    paths: Sequence[str | Path],
    *,
    sigma: float = 25.0,
    neighbor_sigma: float = 6.0,
    min_center_neighbor_ratio: float = 2.0,
    min_frames: int | None = None,
    filter_size: int = 3,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    """Build a fixed sensor-coordinate hot-pixel mask from a sequence."""

    mask, _diagnostic = _build_persistent_hot_pixel_mask_with_diagnostic(
        paths,
        sigma=sigma,
        neighbor_sigma=neighbor_sigma,
        min_center_neighbor_ratio=min_center_neighbor_ratio,
        min_frames=min_frames,
        filter_size=filter_size,
        diagnostic_dir=None,
        progress_callback=progress_callback,
    )
    return mask


def _build_persistent_hot_pixel_mask_with_diagnostic(
    paths: Sequence[str | Path],
    *,
    sigma: float = 25.0,
    neighbor_sigma: float = 6.0,
    min_center_neighbor_ratio: float = 2.0,
    min_frames: int | None = None,
    filter_size: int = 3,
    diagnostic_dir: str | Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, HotPixelDiagnostic | None]:
    """Build a fixed hot-pixel mask and optional QA artifacts."""

    if not paths:
        raise ValueError("No images provided")
    detections: list[np.ndarray] = []
    loaded_paths = [Path(path) for path in paths]
    shape: tuple[int, int] | None = None
    total = len(paths)
    for index, path in enumerate(loaded_paths, start=1):
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
        if progress_callback is not None:
            progress_callback(index, total, f"Scanning {Path(path).name}")
    counts = np.sum(np.stack(detections), axis=0)
    required = min_frames if min_frames is not None else max(2, int(np.ceil(len(detections) * 0.8)))
    persistent = counts >= required
    diagnostic = None
    if diagnostic_dir is not None and shape is not None:
        diagnostic = _write_hot_pixel_diagnostics(
            loaded_paths,
            detections,
            persistent,
            required,
            Path(diagnostic_dir),
        )
    return persistent, diagnostic


def _write_hot_pixel_diagnostics(
    paths: Sequence[Path],
    detections: Sequence[np.ndarray],
    persistent: np.ndarray,
    min_frames: int,
    out_dir: Path,
) -> HotPixelDiagnostic:
    out_dir.mkdir(parents=True, exist_ok=True)
    persistent_path = out_dir / f"persistent_mask_{persistent.shape[1]}x{persistent.shape[0]}.png"
    _write_mask_png(persistent, persistent_path)

    transient_union = np.logical_or.reduce([mask & ~persistent for mask in detections])
    transient_path = out_dir / f"transient_union_{persistent.shape[1]}x{persistent.shape[0]}.png"
    _write_mask_png(transient_union, transient_path)

    frame_rows: list[HotPixelFrameDiagnostic] = []
    for path, mask in zip(paths, detections):
        frame_persistent = mask & persistent
        frame_transient = mask & ~persistent
        mask_path = out_dir / f"{path.stem}_hot_pixel_classified.png"
        _write_classified_hot_pixel_png(frame_persistent, frame_transient, mask_path)
        frame_rows.append(
            HotPixelFrameDiagnostic(
                path=path,
                candidate_count=int(mask.sum()),
                persistent_count=int(frame_persistent.sum()),
                transient_count=int(frame_transient.sum()),
                mask_path=mask_path,
            )
        )

    diagnostic = HotPixelDiagnostic(
        shape=tuple(int(value) for value in persistent.shape),
        min_frames=int(min_frames),
        persistent_count=int(persistent.sum()),
        transient_count=int(transient_union.sum()),
        persistent_mask_path=persistent_path,
        transient_mask_path=transient_path,
        frame_diagnostics=frame_rows,
    )
    _write_hot_pixel_summary(out_dir, diagnostic)
    return diagnostic


def _write_hot_pixel_summary(out_dir: Path, diagnostic: HotPixelDiagnostic) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "hot_pixel_summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "frame",
                    "width",
                    "height",
                    "min_frames_for_persistent",
                    "candidate_count",
                    "persistent_hits_in_frame",
                    "transient_hits_in_frame",
                    "persistent_mask_total",
                    "transient_union_total",
                    "mask_png",
                ]
            )
        height, width = diagnostic.shape
        for row in diagnostic.frame_diagnostics:
            writer.writerow(
                [
                    row.path.name,
                    width,
                    height,
                    diagnostic.min_frames,
                    row.candidate_count,
                    row.persistent_count,
                    row.transient_count,
                    diagnostic.persistent_count,
                    diagnostic.transient_count,
                    "" if row.mask_path is None else row.mask_path.name,
                ]
            )


def _write_mask_png(mask: np.ndarray, path: Path) -> None:
    data = np.where(np.asarray(mask, dtype=bool), 0, 255).astype(np.uint8)
    Image.fromarray(data, mode="L").save(path)


def _write_classified_hot_pixel_png(persistent: np.ndarray, transient: np.ndarray, path: Path) -> None:
    rgb = np.full((*persistent.shape, 3), 255, dtype=np.uint8)
    rgb[np.asarray(transient, dtype=bool)] = (150, 150, 150)
    rgb[np.asarray(persistent, dtype=bool)] = (0, 0, 0)
    Image.fromarray(rgb, mode="RGB").save(path)


def detect_isolated_hot_pixels(
    data: np.ndarray,
    *,
    sigma: float = 25.0,
    neighbor_sigma: float = 6.0,
    min_center_neighbor_ratio: float = 2.0,
    filter_size: int = 3,
    include_cold: bool = True,
) -> np.ndarray:
    """Detect isolated bright hot pixels and dark dead pixels."""

    image = np.asarray(data, dtype=np.float32)
    local = median_filter(image, size=filter_size)
    residual = image - local
    robust_sigma = _robust_sigma(residual)
    if robust_sigma <= 0 or not np.isfinite(robust_sigma):
        return np.zeros(image.shape, dtype=bool)
    return _isolated_hot_pixel_mask(
        image,
        local,
        residual,
        robust_sigma,
        sigma=sigma,
        neighbor_sigma=neighbor_sigma,
        min_center_neighbor_ratio=min_center_neighbor_ratio,
        include_cold=include_cold,
    )


def _isolated_hot_pixel_mask(
    image: np.ndarray,
    local: np.ndarray,
    residual: np.ndarray,
    robust_sigma: float,
    *,
    sigma: float,
    neighbor_sigma: float,
    min_center_neighbor_ratio: float,
    include_cold: bool = True,
) -> np.ndarray:

    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    neighbor_max = maximum_filter(image, footprint=footprint, mode="nearest")
    neighbor_residual = neighbor_max - local
    bright_center = residual > sigma * robust_sigma
    quiet_neighbors = neighbor_residual < neighbor_sigma * robust_sigma
    isolated_contrast = image / np.maximum(neighbor_max, 1.0) >= min_center_neighbor_ratio
    hot = bright_center & quiet_neighbors & isolated_contrast
    if not include_cold:
        return hot

    neighbor_min = minimum_filter(image, footprint=footprint, mode="nearest")
    cold_residual = local - image
    cold_neighbor_residual = local - neighbor_min
    cold_center = cold_residual > sigma * robust_sigma
    quiet_cold_neighbors = cold_neighbor_residual < neighbor_sigma * robust_sigma
    isolated_dark_contrast = cold_residual / np.maximum(np.abs(local), 1.0) >= 0.35
    return hot | (cold_center & quiet_cold_neighbors & isolated_dark_contrast)


def apply_hot_pixel_mask(data: np.ndarray, mask: np.ndarray, *, filter_size: int = 3) -> np.ndarray:
    """Replace only the pixels in a supplied mask with local medians."""

    image = np.asarray(data, dtype=np.float32)
    cleaned = image.copy()
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return cleaned.astype(np.float32)
    if filter_size == 3:
        cleaned[mask] = _masked_3x3_median(image, mask)
    else:
        local = median_filter(image, size=filter_size)
        cleaned[mask] = local[mask]
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


def _masked_3x3_median(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    padded_mask = np.pad(mask, 1, mode="edge")
    ys, xs = np.nonzero(mask)
    neighborhoods = np.empty((len(ys), 9), dtype=np.float32)
    valid = np.empty((len(ys), 9), dtype=bool)
    index = 0
    for dy in range(3):
        for dx in range(3):
            neighborhoods[:, index] = padded[ys + dy, xs + dx]
            valid[:, index] = ~padded_mask[ys + dy, xs + dx]
            index += 1
    valid[:, 4] = False
    neighborhoods[~valid] = np.nan
    replacements = np.nanmedian(neighborhoods, axis=1)
    fallback = median_filter(image, size=3)[ys, xs]
    replacements = np.where(np.isfinite(replacements), replacements, fallback)
    return replacements.astype(np.float32)


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
        shape = _image_shape(path)
        groups.setdefault(shape, []).append(path)
    return groups


def _image_shape(path: str | Path) -> tuple[int, int]:
    image_path = Path(path)
    if image_path.suffix.lower() in {".fit", ".fits", ".fts", ".new"}:
        header = fits.getheader(image_path)
        if "NAXIS1" in header and "NAXIS2" in header:
            return int(header["NAXIS2"]), int(header["NAXIS1"])
    return load_image(path).data.shape
