from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
) -> list[AlignedFrame]:
    """Align images to a reference using star-pattern matching."""

    if not paths:
        raise ValueError("No images provided")

    loaded = [load_image(path) for path in paths]
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
    for image in loaded[start:]:
        transform, aligned, footprint, method = _align_one(image.data, reference_image.data)
        rms_error, matched_sources = measure_alignment_error(aligned, reference_image.data)
        frame = AlignedFrame(image, aligned, transform, footprint, method, rms_error, matched_sources)
        result.append(frame)

    if crop_overlap:
        result = crop_to_common_overlap(result)

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
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


def _align_one(data: np.ndarray, reference: np.ndarray) -> tuple[SimilarityTransform, np.ndarray, np.ndarray, str]:
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
