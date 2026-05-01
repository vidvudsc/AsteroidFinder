from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sep


@dataclass(frozen=True)
class Source:
    x: float
    y: float
    flux: float
    a: float
    b: float
    theta: float
    snr: float


def detect_sources(
    data: np.ndarray,
    *,
    sigma: float = 3.0,
    min_area: int = 5,
    max_sources: int | None = 2000,
) -> list[Source]:
    """Detect sources with SEP against a local background model."""

    image = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    bkg = sep.Background(image)
    residual = image - bkg.back()
    objects = sep.extract(residual, sigma, err=bkg.globalrms, minarea=min_area)
    sources = [
        Source(
            x=float(obj["x"]),
            y=float(obj["y"]),
            flux=float(obj["flux"]),
            a=float(obj["a"]),
            b=float(obj["b"]),
            theta=float(obj["theta"]),
            snr=float(obj["peak"] / max(bkg.globalrms, 1e-6)),
        )
        for obj in objects
        if np.isfinite(obj["flux"]) and obj["flux"] > 0
    ]
    sources.sort(key=lambda src: src.flux, reverse=True)
    if max_sources is not None:
        return sources[:max_sources]
    return sources

