from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import astroalign as aa
import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp

from .detection import detect_sources
from .io import AstroImage, load_image, save_fits


@dataclass(frozen=True)
class AlignedFrame:
    image: AstroImage
    data: np.ndarray
    transform: SimilarityTransform | None
    footprint: np.ndarray | None
    method: str = "reference"
    rms_error: float | None = None
    matched_sources: int = 0


def align_images(
    paths: Sequence[str | Path],
    *,
    reference: str | Path | None = None,
    output_dir: str | Path | None = None,
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
            save_fits(frame.data, out_dir / f"{frame.image.path.stem}_aligned.fits", frame.image.header)
    return result


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

    ref_sources = detect_sources(reference, sigma=sigma, max_sources=max_sources)
    src_sources = detect_sources(data, sigma=sigma, max_sources=max_sources)
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
        transform, _ = aa.find_transform(data, reference)
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
    image = np.nan_to_num(np.asarray(data, dtype=np.float32), copy=False)
    median = float(np.median(image))
    std = float(np.std(image))
    if not np.isfinite(std) or std <= 1e-6:
        return image - median
    return ((image - median) / std).astype(np.float32)
