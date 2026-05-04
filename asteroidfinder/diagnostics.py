from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .io import load_image, stretch_to_uint8
from .tracking import Track


def plot_track_diagnostics(
    tracks: Sequence[Track],
    output_dir: str | Path,
    *,
    frame_times_jd: Sequence[float] | None = None,
    prefix: str = "track",
) -> list[Path]:
    """Write one diagnostic PNG per moving-object track.

    Each plot shows image-plane motion, fitted x/y trends, sky-coordinate motion
    when WCS enrichment is available, and image-plane residuals.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("matplotlib is required for track diagnostic plots") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_rows: list[dict[str, str]] = []

    for track_id, track in enumerate(tracks, start=1):
        points = _track_arrays(track, frame_times_jd=frame_times_jd)
        path = out_dir / f"{prefix}_{track_id:03d}_diagnostic.png"
        _plot_one_track(plt, track_id, track, points, path)
        written.append(path)
        summary_rows.append(_summary_row(track_id, track, points, path))

    if summary_rows:
        _write_summary(summary_rows, out_dir / f"{prefix}_diagnostics.csv")
    return written


def write_track_diagnostic_outputs(
    tracks: Sequence[Track],
    output_dir: str | Path,
    *,
    frame_paths: Sequence[str | Path] | None = None,
    frame_times_jd: Sequence[float] | None = None,
    prefix: str = "track",
    cutout_radius: int = 40,
    cutout_scale: int = 4,
) -> list[Path]:
    """Write movement plots plus visual detection products for the tracks.

    Outputs include:
    - one matplotlib movement PNG per track
    - one cutout blink GIF per track when frames are supplied
    - one full-frame PNG overlay with all detected tracks when frames are supplied
    """

    written = plot_track_diagnostics(tracks, output_dir, frame_times_jd=frame_times_jd, prefix=prefix)
    if frame_paths:
        written.extend(
            write_track_cutout_gifs(
                tracks,
                frame_paths,
                output_dir,
                radius=cutout_radius,
                scale=cutout_scale,
                prefix=prefix,
            )
        )
        overlay = write_full_frame_tracks_png(tracks, frame_paths[0], output_dir)
        if overlay is not None:
            written.append(overlay)
    return written


def write_track_cutout_gifs(
    tracks: Sequence[Track],
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    radius: int = 40,
    scale: int = 4,
    prefix: str = "track",
    duration_ms: int = 450,
) -> list[Path]:
    """Write one square blink GIF per detected track."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in frame_paths]
    cache: dict[int, np.ndarray] = {}
    written: list[Path] = []

    for track_id, track in enumerate(tracks, start=1):
        frames: list[Image.Image] = []
        color = _color_for_index(track_id)
        for detection in track.detections:
            if detection.frame_index < 0 or detection.frame_index >= len(paths):
                continue
            preview = cache.get(detection.frame_index)
            if preview is None:
                preview = stretch_to_uint8(load_image(paths[detection.frame_index]).data)
                cache[detection.frame_index] = preview
            cutout = _square_cutout(preview, detection.source.x, detection.source.y, radius)
            image = Image.fromarray(cutout, mode="L").convert("RGB")
            if scale > 1:
                image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            cx = image.width / 2
            cy = image.height / 2
            r = max(7, int(radius * scale * 0.18))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=max(2, scale))
            draw.text((8, 6), f"AF{track_id:05d} f{detection.frame_index}", fill=color)
            frames.append(image)
        if frames:
            path = out_dir / f"{prefix}_{track_id:03d}_cutout.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
            written.append(path)
    return written


def write_full_frame_tracks_png(
    tracks: Sequence[Track],
    frame_path: str | Path,
    output_dir: str | Path,
    *,
    filename: str = "all_detected_tracks.png",
) -> Path | None:
    """Write one full-frame overlay showing every detected moving track."""

    if not tracks:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = stretch_to_uint8(load_image(frame_path).data)
    image = Image.fromarray(preview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    for track_id, track in enumerate(tracks, start=1):
        color = _color_for_index(track_id)
        points = [(det.source.x, det.source.y) for det in track.detections]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for x, y in points:
            r = 12
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        if points:
            x, y = points[0]
            draw.text((x + 16, y - 16), f"AF{track_id:05d}", fill=color)

    path = out_dir / filename
    image.save(path)
    return path


def _track_arrays(track: Track, *, frame_times_jd: Sequence[float] | None) -> dict[str, np.ndarray | float | None | str]:
    frames = np.array([det.frame_index for det in track.detections], dtype=np.float64)
    xs = np.array([det.source.x for det in track.detections], dtype=np.float64)
    ys = np.array([det.source.y for det in track.detections], dtype=np.float64)

    if frame_times_jd is not None:
        times = np.array([frame_times_jd[det.frame_index] for det in track.detections], dtype=np.float64)
        t = (times - times[0]) * 1440.0
        time_label = "minutes from first detection"
        time_unit = "min"
    else:
        t = frames - frames[0]
        time_label = "frames from first detection"
        time_unit = "frame"

    vx, x0 = np.polyfit(t, xs, 1)
    vy, y0 = np.polyfit(t, ys, 1)
    fit_x = vx * t + x0
    fit_y = vy * t + y0
    residual_px = np.hypot(xs - fit_x, ys - fit_y)

    ra_values = [det.ra_deg for det in track.detections]
    dec_values = [det.dec_deg for det in track.detections]
    if all(value is not None for value in ra_values) and all(value is not None for value in dec_values):
        ra = np.array([float(value) for value in ra_values], dtype=np.float64)
        dec = np.array([float(value) for value in dec_values], dtype=np.float64)
        dec_ref = np.deg2rad(np.nanmedian(dec))
        ra_offset = (ra - ra[0]) * np.cos(dec_ref) * 3600.0
        dec_offset = (dec - dec[0]) * 3600.0
        vra, ra0 = np.polyfit(t, ra_offset, 1)
        vdec, dec0 = np.polyfit(t, dec_offset, 1)
        sky_speed = float(np.hypot(vra, vdec))
        pa = float((np.degrees(np.arctan2(vra, vdec)) + 360.0) % 360.0)
        fit_ra = vra * t + ra0
        fit_dec = vdec * t + dec0
    else:
        ra = dec = ra_offset = dec_offset = fit_ra = fit_dec = None
        sky_speed = pa = None

    pixel_speed = float(np.hypot(vx, vy))
    return {
        "frames": frames,
        "t": t,
        "time_label": time_label,
        "time_unit": time_unit,
        "x": xs,
        "y": ys,
        "fit_x": fit_x,
        "fit_y": fit_y,
        "residual_px": residual_px,
        "vx": float(vx),
        "vy": float(vy),
        "pixel_speed": pixel_speed,
        "ra": ra,
        "dec": dec,
        "ra_offset": ra_offset,
        "dec_offset": dec_offset,
        "fit_ra": fit_ra,
        "fit_dec": fit_dec,
        "sky_speed": sky_speed,
        "pa": pa,
    }


def _square_cutout(data: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    output = np.zeros((size, size), dtype=np.uint8)
    cx = int(round(x))
    cy = int(round(y))
    src_x0 = max(0, cx - radius)
    src_x1 = min(data.shape[1], cx + radius + 1)
    src_y0 = max(0, cy - radius)
    src_y1 = min(data.shape[0], cy + radius + 1)
    dst_x0 = src_x0 - (cx - radius)
    dst_y0 = src_y0 - (cy - radius)
    output[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = data[src_y0:src_y1, src_x0:src_x1]
    return output


def _color_for_index(index: int) -> tuple[int, int, int]:
    colors = [
        (56, 189, 248),
        (52, 211, 153),
        (251, 191, 36),
        (248, 113, 113),
        (167, 139, 250),
        (244, 114, 182),
        (45, 212, 191),
        (250, 204, 21),
    ]
    return colors[(index - 1) % len(colors)]


def _plot_one_track(plt: object, track_id: int, track: Track, points: dict[str, object], path: Path) -> None:
    t = points["t"]
    xs = points["x"]
    ys = points["y"]
    fit_x = points["fit_x"]
    fit_y = points["fit_y"]
    residual = points["residual_px"]
    time_label = str(points["time_label"])
    time_unit = str(points["time_unit"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(_track_title(track_id, track, points), fontsize=13)

    ax = axes[0, 0]
    ax.plot(xs, ys, "o", color="#1f77b4", label="detections")
    ax.plot(fit_x, fit_y, "-", color="#d62728", label="linear fit")
    for frame, x, y in zip(points["frames"], xs, ys):
        ax.annotate(str(int(frame)), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title("image-plane motion")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, xs, "o", color="#1f77b4", label="x")
    ax.plot(t, fit_x, "-", color="#1f77b4", alpha=0.65)
    ax.plot(t, ys, "o", color="#ff7f0e", label="y")
    ax.plot(t, fit_y, "-", color="#ff7f0e", alpha=0.65)
    ax.set_xlabel(time_label)
    ax.set_ylabel("pixels")
    ax.set_title(f"x/y trend: vx={points['vx']:.3f}, vy={points['vy']:.3f} px/{time_unit}")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    if points["ra_offset"] is not None:
        ax.plot(t, points["ra_offset"], "o", color="#2ca02c", label="RA*cos(dec)")
        ax.plot(t, points["fit_ra"], "-", color="#2ca02c", alpha=0.65)
        ax.plot(t, points["dec_offset"], "o", color="#9467bd", label="Dec")
        ax.plot(t, points["fit_dec"], "-", color="#9467bd", alpha=0.65)
        ax.set_ylabel("arcsec from first detection")
        ax.set_title(f"sky trend: {points['sky_speed']:.3f} arcsec/{time_unit}, PA={points['pa']:.1f} deg")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No WCS RA/Dec on track", ha="center", va="center")
        ax.set_title("sky trend")
    ax.set_xlabel(time_label)

    ax = axes[1, 1]
    ax.plot(t, residual, "o-", color="#d62728")
    ax.set_xlabel(time_label)
    ax.set_ylabel("residual [px]")
    ax.set_title(f"fit residual: mean={np.mean(residual):.3f}px, max={np.max(residual):.3f}px")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def _track_title(track_id: int, track: Track, points: dict[str, object]) -> str:
    parts = [
        f"Track {track_id}",
        f"{len(track.detections)} detections",
        f"{points['pixel_speed']:.3f} px/{points['time_unit']}",
    ]
    if points["sky_speed"] is not None:
        parts.append(f"{points['sky_speed']:.3f} arcsec/{points['time_unit']}")
    if track.position_angle_deg is not None:
        parts.append(f"PA {track.position_angle_deg:.1f} deg")
    return " | ".join(parts)


def _summary_row(track_id: int, track: Track, points: dict[str, object], path: Path) -> dict[str, str]:
    residual = points["residual_px"]
    return {
        "track_id": str(track_id),
        "detections": str(len(track.detections)),
        "velocity_x": f"{points['vx']:.6f}",
        "velocity_y": f"{points['vy']:.6f}",
        "pixel_speed": f"{points['pixel_speed']:.6f}",
        "pixel_speed_unit": f"px/{points['time_unit']}",
        "sky_speed": "" if points["sky_speed"] is None else f"{points['sky_speed']:.6f}",
        "sky_speed_unit": "" if points["sky_speed"] is None else f"arcsec/{points['time_unit']}",
        "position_angle_deg": "" if points["pa"] is None else f"{points['pa']:.4f}",
        "mean_residual_px": f"{np.mean(residual):.6f}",
        "max_residual_px": f"{np.max(residual):.6f}",
        "png": path.name,
    }


def _write_summary(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
