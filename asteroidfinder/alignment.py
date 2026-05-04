from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import csv

import astroalign as aa
import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates, median_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp

from .detection import detect_sources
from .io import AstroImage, load_image, save_fits


_FRAME_METADATA_KEYS = (
    "DATE-OBS",
    "DATEOBS",
    "DATE",
    "TIME-OBS",
    "UT",
    "UTC-OBS",
    "OBSJD",
    "JD",
    "JULDATE",
    "MJD-OBS",
    "MJD",
    "EXPTIME",
    "EXPOSURE",
    "FILTER",
    "FILTERID",
    "OBJECT",
    "OBJCTRA",
    "OBJCTDEC",
    "RA",
    "DEC",
    "IMAGETYP",
    "XBINNING",
    "YBINNING",
    "GAIN",
    "OFFSET",
    "AIRMASS",
)


@dataclass(frozen=True)
class AlignedFrame:
    image: AstroImage
    data: np.ndarray
    transform: SimilarityTransform | None
    footprint: np.ndarray | None
    method: str = "reference"
    rms_error: float | None = None
    matched_sources: int = 0
    origin_x: int = 0
    origin_y: int = 0


def align_images(
    paths: Sequence[str | Path],
    *,
    reference: str | Path | None = None,
    output_dir: str | Path | None = None,
    qa_path: str | Path | None = None,
    crop_overlap: bool = False,
    prefer_translation: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[AlignedFrame]:
    """Align images to a reference using star-pattern matching."""

    if not paths:
        raise ValueError("No images provided")

    total = max(1, len(paths))
    loaded = [load_image(path) for path in paths]
    if progress_callback is not None:
        progress_callback(1, total, "Loaded reference frame")
    reference_image = load_image(reference) if reference is not None else loaded[0]
    result = [
        AlignedFrame(
            reference_image,
            reference_image.data.copy(),
            None,
            np.ones(reference_image.data.shape, dtype=bool),
        )
    ]

    start = 1 if reference is None else 0
    done = 1
    for image in loaded[start:]:
        if progress_callback is not None:
            progress_callback(done, total, f"Aligning {image.path.name}")
        try:
            transform, aligned, footprint, method = _align_by_wcs_if_possible(image, reference_image)
        except ValueError:
            transform, aligned, footprint, method = _align_one(image.data, reference_image.data, prefer_translation=prefer_translation)
        rms_error, matched_sources = measure_alignment_error(aligned, reference_image.data)
        frame = AlignedFrame(image, aligned, transform, footprint, method, rms_error, matched_sources)
        result.append(frame)
        done += 1
        if progress_callback is not None:
            progress_callback(done, total, f"Aligned {image.path.name}")

    if crop_overlap:
        result = crop_to_common_overlap(result)
        if progress_callback is not None:
            progress_callback(done, total, "Cropped common overlap")

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(total, total, "Writing aligned FITS files")
        for frame in result:
            header = _aligned_output_header(
                frame.image.header,
                reference_image.header,
                origin_x=frame.origin_x,
                origin_y=frame.origin_y,
            )
            save_fits(frame.data, out_dir / f"{frame.image.path.stem}_aligned.fits", header)
        write_alignment_qa(result, qa_path if qa_path is not None else out_dir / "alignment_qa.csv")
    elif qa_path is not None:
        write_alignment_qa(result, qa_path)
    return result


def _aligned_output_header(
    frame_header: object | None,
    reference_header: object | None,
    *,
    origin_x: int = 0,
    origin_y: int = 0,
) -> object | None:
    """Use reference WCS for aligned pixels while preserving per-frame metadata."""

    if reference_header is None:
        header = frame_header.copy() if hasattr(frame_header, "copy") else None
        return _shift_header_origin(header, origin_x=origin_x, origin_y=origin_y)
    header = reference_header.copy()
    if frame_header is None:
        return _shift_header_origin(header, origin_x=origin_x, origin_y=origin_y)
    for key in _FRAME_METADATA_KEYS:
        if key not in frame_header:
            continue
        header[key] = frame_header[key]
        try:
            header.comments[key] = frame_header.comments[key]
        except Exception:
            pass
    return _shift_header_origin(header, origin_x=origin_x, origin_y=origin_y)


def _shift_header_origin(header: object | None, *, origin_x: int, origin_y: int) -> object | None:
    if header is None or (origin_x == 0 and origin_y == 0):
        return header
    if "CRPIX1" in header:
        header["CRPIX1"] = float(header["CRPIX1"]) - origin_x
    if "CRPIX2" in header:
        header["CRPIX2"] = float(header["CRPIX2"]) - origin_y
    header["AFCROPX"] = (int(origin_x), "AsteroidFinder crop origin x")
    header["AFCROPY"] = (int(origin_y), "AsteroidFinder crop origin y")
    return header


def write_alignment_qa(frames: Sequence[AlignedFrame], path: str | Path) -> Path:
    """Write per-frame alignment quality metrics to CSV."""

    qa_path = Path(path)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "method", "matched_sources", "rms_error_px", "quality"])
        for index, frame in enumerate(frames):
            rms = frame.rms_error
            if index == 0:
                quality = "reference"
            elif rms is None or frame.matched_sources == 0:
                quality = "unknown"
            elif rms <= 1.0:
                quality = "good"
            elif rms <= 2.5:
                quality = "warning"
            else:
                quality = "bad"
            writer.writerow(
                [
                    frame.image.path.name,
                    frame.method,
                    frame.matched_sources,
                    "" if rms is None else f"{rms:.4f}",
                    quality,
                ]
            )
    return qa_path


def crop_to_common_overlap(frames: Sequence[AlignedFrame]) -> list[AlignedFrame]:
    """Crop aligned frames to the largest rectangle valid in every frame."""

    if not frames:
        return []
    masks = [frame.footprint if frame.footprint is not None else np.ones(frame.data.shape, dtype=bool) for frame in frames]
    common = np.logical_and.reduce(masks)
    rows = np.where(common.any(axis=1))[0]
    cols = np.where(common.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Aligned frames have no common valid overlap")
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    cropped: list[AlignedFrame] = []
    for frame in frames:
        footprint = None if frame.footprint is None else frame.footprint[y0:y1, x0:x1]
        cropped.append(
            AlignedFrame(
                frame.image,
                frame.data[y0:y1, x0:x1],
                frame.transform,
                footprint,
                frame.method,
                frame.rms_error,
                frame.matched_sources,
                frame.origin_x + x0,
                frame.origin_y + y0,
            )
        )
    return cropped


def stack_images(frames: Sequence[AlignedFrame] | Sequence[np.ndarray], *, method: str = "median") -> np.ndarray:
    """Stack aligned images with median, mean, or sum combination."""

    arrays = [frame.data if isinstance(frame, AlignedFrame) else np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("No frames to stack")
    cube = np.stack(arrays).astype(np.float32)
    if method == "median":
        return np.nanmedian(cube, axis=0).astype(np.float32)
    if method == "mean":
        return np.nanmean(cube, axis=0).astype(np.float32)
    if method == "sum":
        return np.nansum(cube, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported stack method: {method}")


def measure_alignment_error(
    data: np.ndarray,
    reference: np.ndarray,
    *,
    sigma: float = 5.0,
    match_radius: float = 5.0,
    max_sources: int = 300,
) -> tuple[float | None, int]:
    """Measure residual star-coordinate error between an aligned image and reference."""

    ref_sources = detect_sources(_hot_pixel_suppressed(reference), sigma=sigma, max_sources=max_sources)
    src_sources = detect_sources(_hot_pixel_suppressed(data), sigma=sigma, max_sources=max_sources)
    if not ref_sources or not src_sources:
        return None, 0
    ref_points = np.array([(src.x, src.y) for src in ref_sources], dtype=np.float32)
    src_points = np.array([(src.x, src.y) for src in src_sources], dtype=np.float32)
    distances = []
    for x, y in src_points:
        delta = ref_points - (x, y)
        dist = np.hypot(delta[:, 0], delta[:, 1])
        best = float(np.min(dist))
        if best <= match_radius:
            distances.append(best)
    if not distances:
        return None, 0
    return float(np.sqrt(np.mean(np.square(distances)))), len(distances)


def _align_one(
    data: np.ndarray,
    reference: np.ndarray,
    *,
    prefer_translation: bool = True,
) -> tuple[SimilarityTransform, np.ndarray, np.ndarray, str]:
    if prefer_translation and data.shape == reference.shape:
        try:
            return _align_by_phase(data, reference)
        except ValueError:
            pass
    try:
        source_points = _alignment_control_points(data)
        reference_points = _alignment_control_points(reference)
        transform, _ = aa.find_transform(source_points, reference_points)
        aligned = warp(
            data,
            inverse_map=transform.inverse,
            output_shape=reference.shape,
            preserve_range=True,
            order=3,
            cval=np.nan,
        ).astype(np.float32)
        footprint = warp(
            np.ones(data.shape, dtype=np.float32),
            inverse_map=transform.inverse,
            output_shape=reference.shape,
            preserve_range=True,
            order=0,
            cval=0.0,
        ) > 0.5
        return transform, np.nan_to_num(aligned, nan=0.0), footprint, "astroalign"
    except (aa.MaxIterError, ValueError):
        if data.shape != reference.shape:
            raise
        shift, _, _ = phase_cross_correlation(reference, data, upsample_factor=10)
        aligned = ndi_shift(data, shift=shift, order=3, mode="constant", cval=0.0).astype(np.float32)
        footprint = ndi_shift(np.ones(data.shape, dtype=np.float32), shift=shift, order=0, mode="constant", cval=0.0) > 0.5
        transform = SimilarityTransform(translation=(float(shift[1]), float(shift[0])))
        return transform, aligned, footprint, "phase-correlation"


def _align_by_wcs_if_possible(image: AstroImage, reference: AstroImage) -> tuple[SimilarityTransform | None, np.ndarray, np.ndarray, str]:
    if image.header is None or reference.header is None:
        raise ValueError("WCS alignment requires FITS headers")
    source_wcs = WCS(image.header)
    reference_wcs = WCS(reference.header)
    if not source_wcs.has_celestial or not reference_wcs.has_celestial:
        raise ValueError("WCS alignment requires celestial WCS")
    aligned, footprint = _reproject_to_reference_wcs(image.data, source_wcs, reference_wcs, reference.data.shape)
    return None, aligned, footprint, "wcs-reproject"


def _reproject_to_reference_wcs(
    data: np.ndarray,
    source_wcs: WCS,
    reference_wcs: WCS,
    output_shape: tuple[int, int],
    *,
    chunk_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_shape
    aligned = np.zeros(output_shape, dtype=np.float32)
    footprint = np.zeros(output_shape, dtype=bool)
    x_coords = np.arange(width, dtype=np.float64)
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float64), x_coords, indexing="ij")
        ra, dec = reference_wcs.pixel_to_world_values(xx, yy)
        src_x, src_y = source_wcs.world_to_pixel_values(ra, dec)
        valid = (
            np.isfinite(src_x)
            & np.isfinite(src_y)
            & (src_x >= 0)
            & (src_y >= 0)
            & (src_x <= data.shape[1] - 1)
            & (src_y <= data.shape[0] - 1)
        )
        sampled = map_coordinates(
            np.asarray(data, dtype=np.float32),
            [src_y, src_x],
            order=1,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)
        sampled[~valid] = 0.0
        aligned[y0:y1] = sampled
        footprint[y0:y1] = valid
    return aligned, footprint


def _align_by_phase(data: np.ndarray, reference: np.ndarray) -> tuple[SimilarityTransform, np.ndarray, np.ndarray, str]:
    factor = _phase_downsample_factor(data.shape)
    ref_small = _downsample_for_phase(reference, factor)
    data_small = _downsample_for_phase(data, factor)
    shift, _error, _phase = phase_cross_correlation(ref_small, data_small, upsample_factor=20)
    full_shift = np.asarray(shift, dtype=np.float32) * factor
    if not np.all(np.isfinite(full_shift)):
        raise ValueError("phase correlation produced a non-finite shift")
    aligned = ndi_shift(data, shift=full_shift, order=1, mode="constant", cval=0.0).astype(np.float32)
    footprint = ndi_shift(np.ones(data.shape, dtype=np.float32), shift=full_shift, order=0, mode="constant", cval=0.0) > 0.5
    transform = SimilarityTransform(translation=(float(full_shift[1]), float(full_shift[0])))
    return transform, aligned, footprint, "phase-correlation-fast"


def _phase_downsample_factor(shape: tuple[int, ...]) -> int:
    if len(shape) < 2:
        return 1
    longest = max(shape[-2:])
    if longest >= 5000:
        return 4
    if longest >= 2500:
        return 2
    return 1


def _downsample_for_phase(data: np.ndarray, factor: int) -> np.ndarray:
    image = np.asarray(data, dtype=np.float32)
    if factor <= 1:
        return _normalize_for_phase(image)
    height = image.shape[0] // factor * factor
    width = image.shape[1] // factor * factor
    trimmed = image[:height, :width]
    downsampled = trimmed.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3)).astype(np.float32)
    return _normalize_for_phase(downsampled)


def _normalize_for_phase(data: np.ndarray) -> np.ndarray:
    image = _hot_pixel_suppressed(data)
    median = float(np.median(image))
    std = float(np.std(image))
    if not np.isfinite(std) or std <= 1e-6:
        return image - median
    return ((image - median) / std).astype(np.float32)


def _alignment_control_points(data: np.ndarray, *, max_sources: int = 80) -> np.ndarray:
    cleaned = _hot_pixel_suppressed(data)
    sources = detect_sources(cleaned, sigma=5.0, min_area=5, max_sources=max_sources)
    points = [(source.x, source.y) for source in sources if source.a > 0 and source.b > 0]
    if len(points) < 3:
        raise ValueError("Not enough star-like control points for alignment")
    return np.asarray(points, dtype=np.float32)


def _hot_pixel_suppressed(data: np.ndarray, *, sigma: float = 10.0) -> np.ndarray:
    image = np.nan_to_num(np.asarray(data, dtype=np.float32), copy=True)
    local = median_filter(image, size=3)
    residual = image - local
    mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
    robust_sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(residual))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        return image
    mask = residual > sigma * robust_sigma
    if np.any(mask):
        image[mask] = local[mask]
    return image.astype(np.float32)
