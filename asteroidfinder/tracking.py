from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .alignment import AlignedFrame, align_images
from .detection import Source, detect_sources
from .photometry import Photometry, aperture_photometry
from .wcs import image_wcs, pixel_to_sky


@dataclass(frozen=True)
class TrackDetection:
    frame_index: int
    source: Source
    photometry: Photometry | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None


@dataclass(frozen=True)
class Track:
    detections: tuple[TrackDetection, ...]
    velocity_x: float
    velocity_y: float
    score: float
    angular_rate_arcsec_per_frame: float | None = None
    position_angle_deg: float | None = None


def track_moving_objects(
    paths: Sequence[str | Path],
    *,
    sigma: float = 4.0,
    stationary_radius: float = 2.0,
    link_radius: float = 8.0,
    min_detections: int = 3,
    assume_aligned: bool = False,
    max_sources: int | None = 500,
) -> list[Track]:
    """Find moving-object candidates after stellar alignment.

    Frames are first aligned to the first image. Sources repeated at the same
    aligned coordinates are treated as stationary stars and removed. Remaining
    transient detections are linked by approximately constant pixel velocity.
    """

    if assume_aligned:
        from .io import load_image

        loaded = [load_image(path) for path in paths]
        aligned = [AlignedFrame(image, image.data, None, None, method="prealigned") for image in loaded]
    else:
        aligned = align_images(paths)
    return track_aligned_frames(
        aligned,
        sigma=sigma,
        stationary_radius=stationary_radius,
        link_radius=link_radius,
        min_detections=min_detections,
        max_sources=max_sources,
    )


def track_aligned_frames(
    aligned: Sequence[AlignedFrame],
    *,
    sigma: float = 4.0,
    stationary_radius: float = 2.0,
    link_radius: float = 8.0,
    min_detections: int = 3,
    remove_stationary_sources: bool = True,
    max_sources: int | None = 500,
) -> list[Track]:
    """Find moving-object tracks in frames that are already on a common pixel grid."""

    detections = [detect_sources(frame.data, sigma=sigma, max_sources=max_sources) for frame in aligned]
    if remove_stationary_sources:
        detections = _remove_stationary(detections, radius=stationary_radius)
    tracks = _link_tracks(detections, link_radius=link_radius, min_detections=min_detections)
    tracks = _enrich_tracks(tracks, aligned)
    tracks.sort(key=lambda item: (len(item.detections), item.score), reverse=True)
    return tracks


def _remove_stationary(detections: list[list[Source]], *, radius: float) -> list[list[Source]]:
    if len(detections) < 2:
        return detections
    all_points = np.array([(src.x, src.y) for frame in detections for src in frame], dtype=np.float32)
    if len(all_points) == 0:
        return detections
    tree = cKDTree(all_points)
    filtered: list[list[Source]] = []
    for frame in detections:
        keep = []
        for src in frame:
            neighbors = tree.query_ball_point((src.x, src.y), radius)
            if len(neighbors) <= 1:
                keep.append(src)
        filtered.append(keep)
    return filtered


def _link_tracks(detections: list[list[Source]], *, link_radius: float, min_detections: int) -> list[Track]:
    if len(detections) < min_detections:
        return []
    frame_trees: list[cKDTree | None] = []
    for frame in detections:
        points = np.array([(src.x, src.y) for src in frame], dtype=np.float32)
        frame_trees.append(cKDTree(points) if len(points) else None)
    tracks: list[Track] = []
    max_candidate_tracks = 5000
    for first_index, first_frame in enumerate(detections[:-1]):
        for first in first_frame:
            candidates: list[TrackDetection] = [TrackDetection(first_index, first)]
            last = first
            for frame_index in range(first_index + 1, len(detections)):
                frame = detections[frame_index]
                tree = frame_trees[frame_index]
                if not frame or tree is None:
                    continue
                predicted = _predict(candidates, frame_index)
                distance, source_index = tree.query(predicted, distance_upper_bound=link_radius)
                if not np.isfinite(distance):
                    if len(candidates) == 1:
                        predicted = (last.x, last.y)
                    continue
                best = frame[int(source_index)]
                if distance <= link_radius:
                    candidates.append(TrackDetection(frame_index, best))
                    last = best
                elif len(candidates) == 1:
                    predicted = (last.x, last.y)
            if len(candidates) >= min_detections:
                tracks.append(_make_track(candidates))
                if len(tracks) >= max_candidate_tracks:
                    return _deduplicate_tracks(tracks)
    return _deduplicate_tracks(tracks)


def _predict(track: list[TrackDetection], frame_index: int) -> tuple[float, float]:
    if len(track) < 2:
        return track[-1].source.x, track[-1].source.y
    i0, p0 = track[0].frame_index, track[0].source
    i1, p1 = track[-1].frame_index, track[-1].source
    dt = max(i1 - i0, 1)
    vx = (p1.x - p0.x) / dt
    vy = (p1.y - p0.y) / dt
    step = frame_index - i1
    return p1.x + vx * step, p1.y + vy * step


def _make_track(points: list[TrackDetection]) -> Track:
    indexes = np.array([point.frame_index for point in points], dtype=np.float32)
    xs = np.array([point.source.x for point in points], dtype=np.float32)
    ys = np.array([point.source.y for point in points], dtype=np.float32)
    vx, x0 = np.polyfit(indexes, xs, 1)
    vy, y0 = np.polyfit(indexes, ys, 1)
    residual = np.mean(np.hypot(xs - (vx * indexes + x0), ys - (vy * indexes + y0)))
    score = float(len(points) / (1.0 + residual))
    return Track(tuple(points), float(vx), float(vy), score)


def _enrich_tracks(tracks: Sequence[Track], aligned_frames: Sequence[AlignedFrame]) -> list[Track]:
    wcs_by_frame = [image_wcs(frame.image) for frame in aligned_frames]
    return [_enrich_track(track, aligned_frames, wcs_by_frame) for track in tracks]


def _enrich_track(track: Track, aligned_frames: Sequence[AlignedFrame], wcs_by_frame: Sequence[object] | None = None) -> Track:
    enriched: list[TrackDetection] = []
    sky_points: list[tuple[int, float, float]] = []
    for detection in track.detections:
        frame = aligned_frames[detection.frame_index]
        phot = aperture_photometry(frame.data, detection.source)
        wcs = wcs_by_frame[detection.frame_index] if wcs_by_frame is not None else image_wcs(frame.image)
        x_for_wcs = detection.source.x
        y_for_wcs = detection.source.y
        if frame.transform is not None:
            x_for_wcs, y_for_wcs = map(
                float,
                frame.transform.inverse(np.array([[detection.source.x, detection.source.y]], dtype=float))[0],
            )
        ra, dec = pixel_to_sky(wcs, x_for_wcs, y_for_wcs)
        if ra is not None and dec is not None:
            sky_points.append((detection.frame_index, ra, dec))
        enriched.append(
            TrackDetection(
                detection.frame_index,
                detection.source,
                photometry=phot,
                ra_deg=ra,
                dec_deg=dec,
            )
        )
    angular_rate, pa = _sky_motion(sky_points)
    return Track(tuple(enriched), track.velocity_x, track.velocity_y, track.score, angular_rate, pa)


def _sky_motion(points: list[tuple[int, float, float]]) -> tuple[float | None, float | None]:
    if len(points) < 2:
        return None, None
    i0, ra0, dec0 = points[0]
    i1, ra1, dec1 = points[-1]
    dt = max(i1 - i0, 1)
    dec_mid = np.deg2rad((dec0 + dec1) / 2.0)
    dra_arcsec = (ra1 - ra0) * np.cos(dec_mid) * 3600.0
    ddec_arcsec = (dec1 - dec0) * 3600.0
    rate = float(np.hypot(dra_arcsec, ddec_arcsec) / dt)
    pa = float((np.degrees(np.arctan2(dra_arcsec, ddec_arcsec)) + 360.0) % 360.0)
    return rate, pa


def _deduplicate_tracks(tracks: list[Track]) -> list[Track]:
    tracks = sorted(tracks, key=lambda item: (len(item.detections), item.score), reverse=True)
    unique: list[Track] = []
    for track in tracks:
        key = _track_key(track)
        if any(key.issubset(_track_key(existing)) for existing in unique):
            continue
        unique.append(track)
    return unique


def _track_key(track: Track) -> set[tuple[int, int, int]]:
    return {(det.frame_index, round(det.source.x), round(det.source.y)) for det in track.detections}
