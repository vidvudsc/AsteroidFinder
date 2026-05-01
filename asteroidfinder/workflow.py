from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .alignment import AlignedFrame, align_images, stack_images
from .calibration import calibrate_images
from .io import save_fits, save_jpeg
from .tracking import Track, track_moving_objects


@dataclass(frozen=True)
class AsteroidWorkflowResult:
    aligned_frames: list[AlignedFrame]
    tracks: list[Track]
    stack_path: Path
    difference_paths: list[Path]
    tracks_path: Path
    alignment_report_path: Path


def run_asteroid_workflow(
    paths: Sequence[str | Path],
    *,
    output_dir: str | Path = "asteroid_run",
    master_bias: str | Path | None = None,
    master_dark: str | Path | None = None,
    master_flat: str | Path | None = None,
    hot_sigma: float = 8.0,
    sigma: float = 4.0,
    min_detections: int = 3,
    make_preview: bool = True,
) -> AsteroidWorkflowResult:
    """Run calibration, alignment, stacking, differencing, and tracking."""

    if not paths:
        raise ValueError("No science images provided")
    out_dir = Path(output_dir)
    calibrated_dir = out_dir / "calibrated"
    aligned_dir = out_dir / "aligned"
    diff_dir = out_dir / "difference"
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    calibration_requested = any(value is not None for value in (master_bias, master_dark, master_flat)) or hot_sigma > 0
    working_paths: list[Path]
    if calibration_requested:
        calibrate_images(
            paths,
            output_dir=calibrated_dir,
            master_bias=master_bias,
            master_dark=master_dark,
            master_flat=master_flat,
            hot_sigma=hot_sigma,
        )
        working_paths = sorted(calibrated_dir.glob("*_calibrated.fits"))
    else:
        working_paths = [Path(path) for path in paths]

    aligned_frames = align_images(working_paths, output_dir=aligned_dir)
    stack = stack_images(aligned_frames, method="median")
    stack_path = out_dir / "stack_median.fits"
    save_fits(stack, stack_path)
    if make_preview:
        save_jpeg(stack, out_dir / "stack_median.jpg")

    difference_paths = []
    for frame in aligned_frames:
        diff = frame.data - stack
        diff_path = diff_dir / f"{frame.image.path.stem}_minus_stack.fits"
        save_fits(diff, diff_path, frame.image.header)
        difference_paths.append(diff_path)
        if make_preview:
            save_jpeg(diff, diff_dir / f"{frame.image.path.stem}_minus_stack.jpg")

    tracks = track_moving_objects(working_paths, sigma=sigma, min_detections=min_detections)
    tracks_path = out_dir / "tracks.csv"
    write_tracks_csv(tracks, tracks_path)
    alignment_report_path = out_dir / "alignment_report.csv"
    write_alignment_report(aligned_frames, alignment_report_path)

    return AsteroidWorkflowResult(
        aligned_frames=aligned_frames,
        tracks=tracks,
        stack_path=stack_path,
        difference_paths=difference_paths,
        tracks_path=tracks_path,
        alignment_report_path=alignment_report_path,
    )


def write_alignment_report(frames: Sequence[AlignedFrame], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        handle.write("frame,path,method,rms_error_px,matched_sources\n")
        for index, frame in enumerate(frames):
            rms = "" if frame.rms_error is None else f"{frame.rms_error:.5f}"
            handle.write(f"{index},{frame.image.path},{frame.method},{rms},{frame.matched_sources}\n")
    return output


def write_tracks_csv(tracks: Sequence[Track], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        handle.write(
            "track_id,frame_index,x,y,ra_deg,dec_deg,flux,net_flux,snr,instrumental_mag,"
            "velocity_x,velocity_y,angular_rate_arcsec_per_frame,position_angle_deg,score\n"
        )
        for track_id, track in enumerate(tracks, start=1):
            for detection in track.detections:
                phot = detection.photometry
                row = [
                    track_id,
                    detection.frame_index,
                    f"{detection.source.x:.3f}",
                    f"{detection.source.y:.3f}",
                    _fmt(detection.ra_deg, 8),
                    _fmt(detection.dec_deg, 8),
                    f"{detection.source.flux:.3f}",
                    _fmt(phot.net_flux if phot else None, 3),
                    _fmt(phot.snr if phot else None, 3),
                    _fmt(phot.instrumental_mag if phot else None, 5),
                    f"{track.velocity_x:.5f}",
                    f"{track.velocity_y:.5f}",
                    _fmt(track.angular_rate_arcsec_per_frame, 5),
                    _fmt(track.position_angle_deg, 3),
                    f"{track.score:.5f}",
                ]
                handle.write(",".join(map(str, row)) + "\n")
    return output


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"
