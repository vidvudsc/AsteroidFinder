from __future__ import annotations

import csv
import html
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asteroidfinder.alignment import AlignedFrame, align_images, stack_images
from asteroidfinder.diagnostics import plot_track_diagnostics
from asteroidfinder.io import load_image, save_fits, save_jpeg
from asteroidfinder.known_objects import query_known_objects_for_frames, write_known_objects_csv
from asteroidfinder.platesolve import solve_image
from asteroidfinder.tracking import Track, track_aligned_frames
from asteroidfinder.workflow import write_alignment_report, write_tracks_csv


INPUT_DIR = ROOT / "data" / "Photographica" / "input"
OBSERVATORY = "I41"
COLORS = [
    (255, 80, 80),
    (80, 200, 255),
    (120, 255, 120),
    (255, 210, 70),
    (210, 120, 255),
    (255, 150, 60),
]


def main() -> int:
    base = Path(__file__).resolve().parent
    out = base / "output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    fits_dir = out / "fits"
    images_dir = out / "images"
    tables_dir = out / "tables"
    diagnostics_dir = out / "diagnostics"
    prepared_dir = fits_dir / "prepared"
    aligned_dir = fits_dir / "aligned"
    diff_dir = fits_dir / "difference"
    preview_dir = images_dir / "previews"
    gif_dir = images_dir / "gifs"
    for folder in (prepared_dir, aligned_dir, diff_dir, preview_dir, gif_dir, tables_dir, diagnostics_dir):
        folder.mkdir(parents=True, exist_ok=True)

    paths = sorted(INPUT_DIR.glob("*_cutout_sciimg.fits"))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected 3 ZTF FITS cutouts in {INPUT_DIR}, found {len(paths)}")

    print("preparing ZTF FITS headers...")
    prepared_paths = _prepare_headers(paths, prepared_dir)

    print("checking embedded WCS / plate solution...")
    _write_solve_report(prepared_paths, tables_dir / "plate_solve_report.csv")

    print("querying SkyBoT known objects...")
    known = query_known_objects_for_frames(prepared_paths, location=OBSERVATORY)
    write_known_objects_csv(known, tables_dir / "known_objects.csv")
    _write_known_summary(known, tables_dir / "known_objects_summary.csv")
    print(f"  known object rows: {len(known)}")

    print("aligning frames...")
    aligned = align_images(prepared_paths, output_dir=aligned_dir, crop_overlap=False)
    write_alignment_report(aligned, tables_dir / "alignment_report.csv")

    print("writing previews and blink GIFs...")
    stack = stack_images(aligned, method="median")
    save_fits(stack, fits_dir / "stack_median.fits")
    save_jpeg(stack, preview_dir / "stack_median.jpg")
    for index, frame in enumerate(aligned, start=1):
        save_jpeg(frame.data, preview_dir / f"aligned_{index:03d}.jpg")
        diff = frame.data - stack
        save_fits(diff, diff_dir / f"difference_{index:03d}.fits", frame.image.header)
        save_jpeg(diff, preview_dir / f"difference_{index:03d}.jpg")

    aligned_known = _known_positions_in_aligned_frames(known, aligned)
    _write_aligned_known_csv(aligned_known, tables_dir / "known_objects_aligned_positions.csv")
    _write_blink_gif(aligned, gif_dir / "blink_aligned.gif")
    _write_known_gif(aligned, aligned_known, gif_dir / "known_objects_expected.gif")

    print("tracking moving objects...")
    tracks = track_aligned_frames(aligned, sigma=5.0, stationary_radius=2.0, link_radius=15.0, min_detections=3)
    write_tracks_csv(tracks, tables_dir / "tracks.csv")
    classifications = _classify_tracks(tracks, aligned_known)
    _write_track_classification(classifications, tables_dir / "track_classification.csv")
    _write_track_frame_comparison(tracks, aligned_known, tables_dir / "track_vs_known_objects.csv")
    frame_times = _frame_times_jd(prepared_paths)
    _write_speed_comparison(tracks, classifications, aligned_known, frame_times, tables_dir / "speed_comparison.csv")
    _write_tracks_gif(aligned, tracks, gif_dir / "detected_tracks.gif")
    plot_track_diagnostics(tracks, diagnostics_dir, frame_times_jd=frame_times)
    print(f"  tracks found: {len(tracks)}")

    report = _write_report(out, prepared_paths, known, tracks, classifications)
    print(f"report: {report}")
    return 0


def _prepare_headers(paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for path in paths:
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data
        if "DATE-OBS" not in header and "OBSJD" in header:
            header["DATE-OBS"] = Time(float(header["OBSJD"]), format="jd", scale="utc").isot
        out = output_dir / path.name
        fits.writeto(out, data, header, overwrite=True)
        prepared.append(out)
    return prepared


def _frame_times_jd(paths: list[Path]) -> list[float]:
    times = []
    for path in paths:
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
            if "OBSJD" in header:
                times.append(float(header["OBSJD"]))
            elif "DATE-OBS" in header:
                times.append(float(Time(header["DATE-OBS"], scale="utc").jd))
            else:
                times.append(float(len(times)))
    return times


def _write_solve_report(paths: list[Path], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "method", "has_celestial_wcs", "crval1", "crval2"])
        for frame in paths:
            solution = solve_image(frame)
            writer.writerow([
                frame.name,
                solution.method,
                solution.wcs.has_celestial,
                f"{solution.wcs.wcs.crval[0]:.8f}",
                f"{solution.wcs.wcs.crval[1]:.8f}",
            ])


def _known_positions_in_aligned_frames(objects: list[object], aligned: list[AlignedFrame]) -> list[dict[str, object]]:
    frame_index = {frame.image.path.resolve(): index for index, frame in enumerate(aligned)}
    rows = []
    for obj in objects:
        index = frame_index.get(Path(obj.frame).resolve())
        if index is None:
            continue
        x = float(obj.x)
        y = float(obj.y)
        transform = aligned[index].transform
        if transform is not None:
            x, y = map(float, transform(np.array([[x, y]], dtype=float))[0])
        rows.append(
            {
                "frame_index": index,
                "name": obj.name,
                "number": obj.number,
                "x": x,
                "y": y,
                "v_mag": obj.v_mag,
                "ra_deg": obj.ra_deg,
                "dec_deg": obj.dec_deg,
            }
        )
    return rows


def _classify_tracks(tracks: list[Track], known_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    known_models = _known_motion_models(known_rows)
    rows = []
    for track_id, track in enumerate(tracks, start=1):
        detections = {det.frame_index: det for det in track.detections}
        candidates = []
        for key, model in known_models.items():
            residuals = []
            for frame_index, det in detections.items():
                expected = model["by_frame"].get(frame_index)
                if expected is None:
                    continue
                residuals.append(float(np.hypot(det.source.x - expected[0], det.source.y - expected[1])))
            if len(residuals) < 3:
                continue
            velocity_delta = float(np.hypot(track.velocity_x - model["vx"], track.velocity_y - model["vy"]))
            candidates.append((float(np.mean(residuals)), velocity_delta, key))
        if candidates:
            mean_sep, velocity_delta, key = min(candidates, key=lambda item: (item[0], item[1]))
            matched = mean_sep <= 5.0 and velocity_delta <= 2.5
            rows.append(
                {
                    "track_id": track_id,
                    "status": "matched_known" if matched else "false_positive_candidate",
                    "known_name": key[0],
                    "known_number": key[1],
                    "detections": len(track.detections),
                    "mean_sep_px": mean_sep,
                    "velocity_delta_px_per_frame": velocity_delta,
                }
            )
        else:
            rows.append(
                {
                    "track_id": track_id,
                    "status": "false_positive_candidate",
                    "known_name": "",
                    "known_number": "",
                    "detections": len(track.detections),
                    "mean_sep_px": "",
                    "velocity_delta_px_per_frame": "",
                }
            )
    return rows


def _known_motion_models(known_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in known_rows:
        grouped[(str(row["name"]), str(row["number"]))].append(row)
    models = {}
    for key, rows in grouped.items():
        if len(rows) < 3:
            continue
        rows = sorted(rows, key=lambda item: int(item["frame_index"]))
        idx = np.array([int(row["frame_index"]) for row in rows], dtype=np.float32)
        xs = np.array([float(row["x"]) for row in rows], dtype=np.float32)
        ys = np.array([float(row["y"]) for row in rows], dtype=np.float32)
        vx, _ = np.polyfit(idx, xs, 1)
        vy, _ = np.polyfit(idx, ys, 1)
        models[key] = {
            "vx": float(vx),
            "vy": float(vy),
            "by_frame": {int(row["frame_index"]): (float(row["x"]), float(row["y"])) for row in rows},
        }
    return models


def _write_known_summary(objects: list[object], path: Path) -> None:
    grouped: dict[str, list[object]] = defaultdict(list)
    for obj in objects:
        grouped[obj.name].append(obj)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "number", "frames", "first_v_mag", "first_x", "first_y", "last_x", "last_y"])
        for name, rows in sorted(grouped.items()):
            first = rows[0]
            last = rows[-1]
            writer.writerow([name, first.number, len(rows), _fmt(first.v_mag, 2), f"{first.x:.2f}", f"{first.y:.2f}", f"{last.x:.2f}", f"{last.y:.2f}"])


def _write_aligned_known_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "name", "number", "aligned_x", "aligned_y", "v_mag", "ra_deg", "dec_deg"])
        for row in rows:
            writer.writerow([row["frame_index"], row["name"], row["number"], f"{float(row['x']):.3f}", f"{float(row['y']):.3f}", _fmt(row["v_mag"], 2), f"{float(row['ra_deg']):.8f}", f"{float(row['dec_deg']):.8f}"])


def _write_track_classification(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "status", "known_name", "known_number", "detections", "mean_separation_px", "velocity_delta_px_per_frame"])
        for row in rows:
            writer.writerow([row["track_id"], row["status"], row["known_name"], row["known_number"], row["detections"], _fmt(row["mean_sep_px"], 3), _fmt(row["velocity_delta_px_per_frame"], 4)])


def _write_track_frame_comparison(tracks: list[Track], known_rows: list[dict[str, object]], path: Path) -> None:
    known_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in known_rows:
        known_by_frame[int(row["frame_index"])].append(row)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "frame_index", "detected_x", "detected_y", "nearest_known", "known_x", "known_y", "separation_px"])
        for track_id, track in enumerate(tracks, start=1):
            for det in track.detections:
                nearest = _nearest_known(det.frame_index, det.source.x, det.source.y, known_by_frame)
                if nearest is None:
                    continue
                row, sep = nearest
                writer.writerow([track_id, det.frame_index, f"{det.source.x:.3f}", f"{det.source.y:.3f}", row["name"], f"{float(row['x']):.3f}", f"{float(row['y']):.3f}", f"{sep:.3f}"])


def _write_speed_comparison(
    tracks: list[Track],
    classifications: list[dict[str, object]],
    known_rows: list[dict[str, object]],
    frame_times_jd: list[float],
    path: Path,
) -> None:
    known_by_name: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in known_rows:
        known_by_name[(str(row["name"]), str(row["number"]))].append(row)
    class_by_track = {int(row["track_id"]): row for row in classifications}
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "track_id",
                "known_name",
                "known_number",
                "measured_px_per_min",
                "expected_px_per_min",
                "pixel_speed_delta_px_per_min",
                "measured_arcsec_per_min",
                "expected_arcsec_per_min",
                "sky_speed_delta_arcsec_per_min",
                "measured_pa_deg",
                "expected_pa_deg",
                "pa_delta_deg",
            ]
        )
        for track_id, track in enumerate(tracks, start=1):
            classification = class_by_track.get(track_id, {})
            key = (str(classification.get("known_name", "")), str(classification.get("known_number", "")))
            known_motion = _motion_from_known_rows(known_by_name.get(key, []), frame_times_jd)
            track_motion = _motion_from_track(track, frame_times_jd)
            writer.writerow(
                [
                    track_id,
                    key[0],
                    key[1],
                    _fmt(track_motion["pixel_speed"], 6),
                    _fmt(known_motion.get("pixel_speed"), 6),
                    _fmt(_delta(track_motion["pixel_speed"], known_motion.get("pixel_speed")), 6),
                    _fmt(track_motion["sky_speed"], 6),
                    _fmt(known_motion.get("sky_speed"), 6),
                    _fmt(_delta(track_motion["sky_speed"], known_motion.get("sky_speed")), 6),
                    _fmt(track_motion["pa"], 4),
                    _fmt(known_motion.get("pa"), 4),
                    _fmt(_angle_delta(track_motion["pa"], known_motion.get("pa")), 4),
                ]
            )


def _motion_from_track(track: Track, frame_times_jd: list[float]) -> dict[str, float | None]:
    rows = []
    for det in track.detections:
        rows.append(
            {
                "frame_index": det.frame_index,
                "x": det.source.x,
                "y": det.source.y,
                "ra_deg": det.ra_deg,
                "dec_deg": det.dec_deg,
            }
        )
    return _motion_from_rows(rows, frame_times_jd)


def _motion_from_known_rows(rows: list[dict[str, object]], frame_times_jd: list[float]) -> dict[str, float | None]:
    return _motion_from_rows(rows, frame_times_jd)


def _motion_from_rows(rows: list[dict[str, object]], frame_times_jd: list[float]) -> dict[str, float | None]:
    if len(rows) < 2:
        return {"pixel_speed": None, "sky_speed": None, "pa": None}
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    t = np.array([(frame_times_jd[int(row["frame_index"])] - frame_times_jd[int(rows[0]["frame_index"])]) * 1440.0 for row in rows], dtype=np.float64)
    xs = np.array([float(row["x"]) for row in rows], dtype=np.float64)
    ys = np.array([float(row["y"]) for row in rows], dtype=np.float64)
    vx, _ = np.polyfit(t, xs, 1)
    vy, _ = np.polyfit(t, ys, 1)
    pixel_speed = float(np.hypot(vx, vy))

    ra_values = [row.get("ra_deg") for row in rows]
    dec_values = [row.get("dec_deg") for row in rows]
    if any(value is None or value == "" for value in ra_values + dec_values):
        return {"pixel_speed": pixel_speed, "sky_speed": None, "pa": None}
    ra = np.array([float(value) for value in ra_values], dtype=np.float64)
    dec = np.array([float(value) for value in dec_values], dtype=np.float64)
    dec_ref = np.deg2rad(np.median(dec))
    ra_offset = (ra - ra[0]) * np.cos(dec_ref) * 3600.0
    dec_offset = (dec - dec[0]) * 3600.0
    vra, _ = np.polyfit(t, ra_offset, 1)
    vdec, _ = np.polyfit(t, dec_offset, 1)
    sky_speed = float(np.hypot(vra, vdec))
    pa = float((np.degrees(np.arctan2(vra, vdec)) + 360.0) % 360.0)
    return {"pixel_speed": pixel_speed, "sky_speed": sky_speed, "pa": pa}


def _delta(a: object, b: object) -> float | None:
    if a is None or b is None or a == "" or b == "":
        return None
    return float(a) - float(b)


def _angle_delta(a: object, b: object) -> float | None:
    if a is None or b is None or a == "" or b == "":
        return None
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _nearest_known(frame_index: int, x: float, y: float, known_by_frame: dict[int, list[dict[str, object]]]) -> tuple[dict[str, object], float] | None:
    rows = known_by_frame.get(frame_index, [])
    if not rows:
        return None
    distances = [(row, float(np.hypot(x - float(row["x"]), y - float(row["y"])))) for row in rows]
    return min(distances, key=lambda item: item[1])


def _write_blink_gif(frames: list[AlignedFrame], path: Path) -> None:
    lo, hi = _shared_range([frame.data for frame in frames])
    images = []
    for index, frame in enumerate(frames, start=1):
        image, _ = _display_image(frame.data, lo, hi)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 118, 34), fill=(0, 0, 0))
        draw.text((14, 14), f"Frame {index:02d}", fill=(255, 255, 255))
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=650, loop=0)


def _write_known_gif(frames: list[AlignedFrame], known_rows: list[dict[str, object]], path: Path) -> None:
    lo, hi = _shared_range([frame.data for frame in frames])
    color_by_name = {name: COLORS[index % len(COLORS)] for index, name in enumerate(sorted({str(row["name"]) for row in known_rows}))}
    rows_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in known_rows:
        rows_by_frame[int(row["frame_index"])].append(row)
    images = []
    for index, frame in enumerate(frames):
        image, scale = _display_image(frame.data, lo, hi)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 190, 34), fill=(0, 0, 0))
        draw.text((14, 14), f"SkyBoT frame {index + 1:02d}", fill=(255, 255, 255))
        for row in rows_by_frame.get(index, []):
            color = color_by_name[str(row["name"])]
            x = float(row["x"]) * scale
            y = float(row["y"]) * scale
            r = 12
            width = 4 if str(row["name"]) == "Photographica" else 2
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)
            if str(row["name"]) == "Photographica":
                draw.text((x + r + 4, y - r), str(row["name"]), fill=color)
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=750, loop=0)


def _write_tracks_gif(frames: list[AlignedFrame], tracks: list[Track], path: Path) -> None:
    lo, hi = _shared_range([frame.data for frame in frames])
    detections_by_frame: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for track_id, track in enumerate(tracks, start=1):
        for det in track.detections:
            detections_by_frame[det.frame_index].append((track_id, det.source.x, det.source.y))
    images = []
    for index, frame in enumerate(frames):
        image, scale = _display_image(frame.data, lo, hi)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 160, 34), fill=(0, 0, 0))
        draw.text((14, 14), f"Tracks {index + 1:02d}", fill=(255, 255, 255))
        for track_id, x_raw, y_raw in detections_by_frame.get(index, []):
            color = COLORS[(track_id - 1) % len(COLORS)]
            x = x_raw * scale
            y = y_raw * scale
            r = 14
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
            draw.text((x + r + 4, y - r), f"T{track_id}", fill=color)
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=750, loop=0)


def _write_report(out: Path, paths: list[Path], known: list[object], tracks: list[Track], classifications: list[dict[str, object]]) -> Path:
    path = out / "report.html"
    names = sorted({obj.name for obj in known})
    matched = sum(1 for row in classifications if row["status"] == "matched_known")
    false_pos = sum(1 for row in classifications if row["status"] == "false_positive_candidate")
    speed_rows = _read_csv(out / "tables" / "speed_comparison.csv")
    classification_rows = _read_csv(out / "tables" / "track_classification.csv")
    track_vs_known_rows = _read_csv(out / "tables" / "track_vs_known_objects.csv")
    diagnostics_rows = _read_csv(out / "diagnostics" / "track_diagnostics.csv")
    alignment_rows = _read_csv(out / "tables" / "alignment_report.csv")
    solve_rows = _read_csv(out / "tables" / "plate_solve_report.csv")
    known_summary_rows = _read_csv(out / "tables" / "known_objects_summary.csv")
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AsteroidFinder Demo 2</title>
<style>
body{{margin:0;background:#101214;color:#f2f2f0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.42}}
main{{max-width:1320px;margin:0 auto;padding:32px}} a{{color:#8fd0ff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
.panel{{background:#1b1f23;border:1px solid #30363d;border-radius:8px;padding:14px}}
.metric{{font-size:30px;font-weight:750}} img{{max-width:100%;border:1px solid #30363d;background:#000}}
.muted{{color:#adb5bd}} table{{border-collapse:collapse;width:100%;font-size:13px;background:#15191d}} th,td{{border-bottom:1px solid #30363d;padding:7px 8px;text-align:left}} th{{background:#222830;position:sticky;top:0}} .tablewrap{{max-height:360px;overflow:auto;border:1px solid #30363d;border-radius:8px}} h2{{margin-top:30px}}
</style></head><body><main>
<h1>ZTF Photographica Demo</h1>
<p class="muted">Real ZTF 12 arcmin zr cutouts downloaded with ztf.py. No synthetic objects.</p>
<div class="grid">
<div class="panel"><div class="metric">{len(paths)}</div><div>ZTF FITS frames</div></div>
<div class="panel"><div class="metric">{len(names)}</div><div>SkyBoT known objects in cutout</div></div>
<div class="panel"><div class="metric">{len(tracks)}</div><div>linked moving tracks</div></div>
<div class="panel"><div class="metric">{matched}</div><div>matched to known object</div></div>
<div class="panel"><div class="metric">{false_pos}</div><div>false-positive tracks</div></div>
</div>
<h2>Speed Check</h2>
{_table_html(speed_rows)}
<h2>Track Classification</h2>
{_table_html(classification_rows)}
<h2>Images</h2>
<div class="grid">
<div class="panel"><h3>Aligned blink</h3><img src="images/gifs/blink_aligned.gif"></div>
<div class="panel"><h3>SkyBoT expected objects</h3><img src="images/gifs/known_objects_expected.gif"></div>
<div class="panel"><h3>Detected tracks</h3><img src="images/gifs/detected_tracks.gif"></div>
<div class="panel"><h3>Track diagnostic</h3><img src="diagnostics/track_001_diagnostic.png"></div>
</div>
<h2>Fit Diagnostics</h2>
{_table_html(diagnostics_rows)}
<h2>Frame-by-Frame Match</h2>
{_table_html(track_vs_known_rows)}
<h2>Plate Solve / WCS</h2>
{_table_html(solve_rows)}
<h2>Alignment</h2>
{_table_html(alignment_rows)}
<h2>Known Objects Summary</h2>
{_table_html(known_summary_rows)}
<h2>Files</h2>
<p><a href="tables/plate_solve_report.csv">plate_solve_report.csv</a> · <a href="tables/known_objects.csv">known_objects.csv</a> · <a href="tables/known_objects_summary.csv">known_objects_summary.csv</a> · <a href="tables/known_objects_aligned_positions.csv">known_objects_aligned_positions.csv</a> · <a href="tables/tracks.csv">tracks.csv</a> · <a href="tables/track_classification.csv">track_classification.csv</a> · <a href="tables/speed_comparison.csv">speed_comparison.csv</a> · <a href="tables/track_vs_known_objects.csv">track_vs_known_objects.csv</a> · <a href="tables/alignment_report.csv">alignment_report.csv</a> · <a href="diagnostics/track_diagnostics.csv">track_diagnostics.csv</a></p>
<p class="muted">Known objects found: {html.escape(', '.join(names))}</p>
</main></body></html>""",
        encoding="utf-8",
    )
    return path


def _display_image(data: np.ndarray, lo: float, hi: float, max_size: int = 1000) -> tuple[Image.Image, float]:
    arr = np.clip((np.asarray(data, dtype=np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L").convert("RGB")
    scale = min(max_size / image.width, max_size / image.height, 1.0)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return image, scale


def _shared_range(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(data, dtype=np.float32)[np.isfinite(data)] for data in arrays])
    lo, hi = np.percentile(values, (0.5, 99.7))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _fmt(value: object, digits: int) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _table_html(rows: list[dict[str, str]], *, max_rows: int = 80) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows[:max_rows]:
        body.append("<tr>" + "".join(f"<td>{html.escape(row.get(header, ''))}</td>" for header in headers) + "</tr>")
    note = "" if len(rows) <= max_rows else f"<p class='muted'>Showing {max_rows} of {len(rows)} rows.</p>"
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{note}"


if __name__ == "__main__":
    raise SystemExit(main())
