from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asteroidfinder.alignment import align_images, stack_images
from asteroidfinder.calibration import calibrate_images_with_persistent_hot_pixels
from asteroidfinder.io import load_image, save_fits, save_jpeg
from asteroidfinder.known_objects import (
    forced_photometry_for_known_objects,
    query_known_objects_for_frames,
    write_known_object_photometry_csv,
    write_known_objects_csv,
    write_mpc_observations,
)
from asteroidfinder.platesolve import solve_image
from asteroidfinder.report import generate_html_report
from asteroidfinder.tracking import track_moving_objects
from asteroidfinder.workflow import write_alignment_report, write_tracks_csv

ANNOTATION_COLORS = [
    (255, 80, 80),
    (80, 200, 255),
    (120, 255, 120),
    (255, 210, 70),
    (210, 120, 255),
    (255, 140, 40),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AsteroidFinder demo pipeline on the 6 local frames.")
    parser.add_argument("--data", type=Path, default=Path("data/raw"), help="Input data folder.")
    parser.add_argument("--out", type=Path, default=Path("demo/output"), help="Demo output folder.")
    parser.add_argument("--use-raw", action="store_true", help="Use raw unsolved FITS instead of calibrated luminance FITS.")
    parser.add_argument("--force-astrometry", action="store_true", help="Run astrometry.net even if WCS is already embedded.")
    parser.add_argument("--allow-unsolved", action="store_true", help="Continue with calibrated frames if plate solving fails.")
    parser.add_argument("--timeout", type=int, default=240, help="Per-frame solve-field timeout seconds.")
    parser.add_argument("--index-dir", type=Path, help="Directory containing astrometry.net index-*.fits files.")
    parser.add_argument("--scale-low", type=float, default=1.0, help="Plate solve lower pixel scale arcsec/pixel.")
    parser.add_argument("--scale-high", type=float, default=1.5, help="Plate solve upper pixel scale arcsec/pixel.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Detection sigma for tracking.")
    parser.add_argument("--hot-sigma", type=float, default=25.0, help="Persistent hot-pixel center threshold.")
    parser.add_argument("--hot-neighbor-sigma", type=float, default=6.0, help="Reject hot pixels with bright neighbors.")
    parser.add_argument("--hot-min-ratio", type=float, default=2.0, help="Minimum center/brightest-neighbor ratio.")
    parser.add_argument("--hot-min-frames", type=int, help="Minimum frames a sensor pixel must be hot.")
    parser.add_argument("--min-detections", type=int, default=3)
    parser.add_argument("--skip-known-objects", action="store_true", help="Skip IMCCE SkyBoT known asteroid lookup.")
    parser.add_argument("--skip-forced-known-photometry", action="store_true", help="Skip forced photometry at known-object positions.")
    parser.add_argument("--observatory", default="500", help="SkyBoT observer code. 500 is geocentric.")
    parser.add_argument("--gif-size", type=int, default=1200, help="Maximum blink GIF width/height.")
    parser.add_argument("--keep", action="store_true", help="Keep existing output folder contents.")
    args = parser.parse_args()

    if args.out.exists() and not args.keep:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    science = _select_science_frames(args.data, use_raw=args.use_raw)
    print(f"selected {len(science)} science frames")
    for path in science:
        print(f"  {path}")

    hot_dir = args.out / "01_hot_cleaned"
    solve_dir = args.out / "02_solved"
    align_dir = args.out / "03_aligned"
    preview_dir = args.out / "04_previews"
    diff_dir = args.out / "05_difference"
    hot_mask_dir = args.out / "06_hot_pixel_masks"
    inverted_dir = args.out / "07_inverted"
    for folder in (hot_dir, solve_dir, align_dir, preview_dir, diff_dir, hot_mask_dir, inverted_dir):
        folder.mkdir(parents=True, exist_ok=True)

    print("calibrating / removing hot pixels...")
    cleaned = calibrate_images_with_persistent_hot_pixels(
        science,
        output_dir=hot_dir,
        hot_sigma=args.hot_sigma,
        neighbor_sigma=args.hot_neighbor_sigma,
        min_center_neighbor_ratio=args.hot_min_ratio,
        min_frames=args.hot_min_frames,
    )
    cleaned_paths = [hot_dir / f"{result.image.path.stem}_calibrated.fits" for result in cleaned]
    hot_report = args.out / "hot_pixel_report.csv"
    hot_total = _write_hot_pixel_report(cleaned, hot_report)
    hot_audit = args.out / "hot_pixel_coordinates.csv"
    _write_hot_pixel_coordinate_audit(cleaned, hot_audit)
    _write_hot_pixel_masks(cleaned, hot_mask_dir, max_size=args.gif_size)
    print(f"  replaced {hot_total} hot pixels")
    print(f"  hot pixel report: {hot_report}")
    print(f"  hot pixel coordinate audit: {hot_audit}")

    print("plate solving...")
    solved_paths = []
    for path in cleaned_paths:
        try:
            solution = solve_image(
                path,
                output_dir=solve_dir,
                index_dir=args.index_dir,
                force_astrometry=args.force_astrometry,
                timeout=args.timeout,
                scale_low=args.scale_low,
                scale_high=args.scale_high,
            )
            solved_path = _materialize_solved_frame(path, solution.solved_fits, solve_dir)
            solved_paths.append(solved_path)
            print(f"  {path.name}: {solution.method} -> {solved_path.name}")
        except Exception as exc:
            if not args.allow_unsolved:
                raise
            solved_path = _materialize_solved_frame(path, None, solve_dir)
            solved_paths.append(solved_path)
            print(f"  {path.name}: solve failed, continuing unsolved -> {solved_path.name}")
            print(f"    reason: {exc}")

    known_objects_path = args.out / "known_objects.csv"
    known_objects = []
    if not args.skip_known_objects:
        print("checking known asteroids / solar-system objects in each solved frame...")
        try:
            known_objects = query_known_objects_for_frames(solved_paths, location=args.observatory)
            write_known_objects_csv(known_objects, known_objects_path)
            _write_known_object_summary(known_objects, args.out / "known_objects_summary.csv")
            mpc_path = write_mpc_observations(known_objects, args.out / "mpc_observations.txt")
            print(f"  known objects: {len(known_objects)} -> {known_objects_path}")
            print(f"  MPC-style draft: {mpc_path}")
            if not args.skip_forced_known_photometry:
                forced = forced_photometry_for_known_objects(known_objects)
                forced_path = write_known_object_photometry_csv(forced, args.out / "known_object_forced_photometry.csv")
                print(f"  forced known-object photometry: {len(forced)} -> {forced_path}")
            annotated = _write_known_objects_annotated_gif(solved_paths, known_objects, args.out / "known_objects_annotated.gif", max_size=args.gif_size)
            print(f"  annotated known-object GIF: {annotated}")
        except Exception as exc:
            write_known_objects_csv([], known_objects_path)
            print(f"  known-object lookup failed: {exc}")
            print(f"  wrote empty known-object report: {known_objects_path}")

    print("aligning frames...")
    aligned = align_images(solved_paths, output_dir=align_dir, crop_overlap=True)
    report_path = args.out / "alignment_report.csv"
    write_alignment_report(aligned, report_path)
    print(f"  alignment report: {report_path}")

    print("writing previews, stack, and difference frames...")
    preview_paths = []
    for index, frame in enumerate(aligned, start=1):
        preview_path = preview_dir / f"aligned_{index:03d}.jpg"
        save_jpeg(frame.data, preview_path)
        _save_inverted_jpeg(frame.data, inverted_dir / f"aligned_{index:03d}_inverted.jpg")
        preview_paths.append(preview_path)

    stack = stack_images(aligned, method="median")
    stack_path = args.out / "stack_median.fits"
    save_fits(stack, stack_path)
    save_jpeg(stack, preview_dir / "stack_median.jpg")
    _save_inverted_jpeg(stack, inverted_dir / "stack_median_inverted.jpg")

    for index, frame in enumerate(aligned, start=1):
        diff = frame.data - stack
        save_fits(diff, diff_dir / f"difference_{index:03d}.fits", frame.image.header)
        save_jpeg(diff, preview_dir / f"difference_{index:03d}.jpg")
        _save_inverted_jpeg(diff, inverted_dir / f"difference_{index:03d}_inverted.jpg")

    gif_path = args.out / "blink_aligned.gif"
    _write_blink_gif(aligned, gif_path, max_size=args.gif_size)
    inverted_gif_path = args.out / "blink_aligned_inverted.gif"
    _write_blink_gif(aligned, inverted_gif_path, max_size=args.gif_size, invert=True)
    print(f"  blink GIF: {gif_path}")
    print(f"  inverted blink GIF: {inverted_gif_path}")

    print("tracking moving-object candidates...")
    tracks = track_moving_objects(solved_paths, sigma=args.sigma, min_detections=args.min_detections)
    tracks_path = args.out / "tracks.csv"
    write_tracks_csv(tracks, tracks_path)
    print(f"  tracks: {len(tracks)} -> {tracks_path}")

    report = generate_html_report(args.out)
    print(f"  HTML report: {report}")
    print("done")
    return 0


def _select_science_frames(data_dir: Path, *, use_raw: bool) -> list[Path]:
    if use_raw:
        pattern = "raw-T68-vidvuds1-ASTEROID_SEARCH_early-*-W-120-*.fit"
    else:
        pattern = "calibrated-T68-vidvuds1-ASTEROID_SEARCH_early-*-W-120-*.fit"
    frames = sorted(data_dir.glob(pattern))
    if len(frames) != 6:
        raise FileNotFoundError(f"Expected 6 frames for pattern {pattern}, found {len(frames)} in {data_dir}")
    return frames


def _materialize_solved_frame(input_path: Path, solved_fits: Path | None, solve_dir: Path) -> Path:
    output = solve_dir / f"{input_path.stem}_solved.fits"
    source = solved_fits or input_path
    image = load_image(source)
    save_fits(image.data, output, image.header)
    return output


def _write_hot_pixel_report(results: list[object], path: Path) -> int:
    total = 0
    with path.open("w", newline="") as handle:
        handle.write("frame,path,hot_pixels\n")
        for index, result in enumerate(results, start=1):
            count = int(result.hot_pixel_mask.sum())
            total += count
            handle.write(f"{index},{result.image.path},{count}\n")
    return total


def _write_hot_pixel_masks(results: list[object], output_dir: Path, *, max_size: int) -> None:
    for index, result in enumerate(results, start=1):
        mask = np.where(result.hot_pixel_mask, 255, 0).astype(np.uint8)
        image = Image.fromarray(mask, mode="L")
        image.thumbnail((max_size, max_size), Image.Resampling.NEAREST)
        image.save(output_dir / f"hot_pixels_{index:03d}.png")


def _write_hot_pixel_coordinate_audit(results: list[object], path: Path) -> None:
    if not results:
        return
    ys, xs = np.where(results[0].hot_pixel_mask)
    with path.open("w", newline="") as handle:
        header = ["x", "y", "frames", "median_value", "median_replacement", "median_delta"]
        handle.write(",".join(header) + "\n")
        for y, x in zip(ys, xs):
            values = []
            replacements = []
            for result in results:
                data = result.image.data
                y0, y1 = max(0, y - 1), min(data.shape[0], y + 2)
                x0, x1 = max(0, x - 1), min(data.shape[1], x + 2)
                patch = data[y0:y1, x0:x1].copy()
                patch[y - y0, x - x0] = np.nan
                replacement = float(np.nanmedian(patch))
                values.append(float(data[y, x]))
                replacements.append(replacement)
            median_value = float(np.median(values))
            median_replacement = float(np.median(replacements))
            handle.write(
                f"{x},{y},{len(results)},{median_value:.3f},{median_replacement:.3f},"
                f"{median_value - median_replacement:.3f}\n"
            )


def _write_known_object_summary(objects: list[object], path: Path) -> None:
    grouped: dict[str, list[object]] = {}
    for obj in objects:
        grouped.setdefault(obj.name, []).append(obj)
    with path.open("w", newline="") as handle:
        handle.write("name,number,type,frames,first_v_mag,first_ra_deg,first_dec_deg,last_ra_deg,last_dec_deg\n")
        for name, rows in sorted(grouped.items()):
            first = rows[0]
            last = rows[-1]
            v = "" if first.v_mag is None else f"{first.v_mag:.2f}"
            handle.write(
                f"{name},{first.number},{first.object_type},{len(rows)},{v},"
                f"{first.ra_deg:.8f},{first.dec_deg:.8f},{last.ra_deg:.8f},{last.dec_deg:.8f}\n"
            )


def _write_known_objects_annotated_gif(
    frame_paths: list[Path],
    objects: list[object],
    path: Path,
    *,
    max_size: int,
) -> Path:
    by_frame: dict[Path, list[object]] = {}
    for obj in objects:
        by_frame.setdefault(Path(obj.frame), []).append(obj)
    images = [load_image(frame).data for frame in frame_paths]
    lo, hi = _shared_display_range(images)
    color_by_name = {
        name: ANNOTATION_COLORS[index % len(ANNOTATION_COLORS)]
        for index, name in enumerate(sorted({obj.name for obj in objects}))
    }
    gif_frames = []
    for index, frame_path in enumerate(frame_paths, start=1):
        image_data = load_image(frame_path).data
        stretched = _stretch_with_range(image_data, lo, hi)
        image = Image.fromarray(stretched, mode="L").convert("RGB")
        scale = min(max_size / image.width, max_size / image.height, 1.0)
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 148, 34), fill=(0, 0, 0))
        draw.text((14, 14), f"Frame {index:02d}", fill=(255, 255, 255))
        for obj in by_frame.get(frame_path, []):
            color = color_by_name.get(obj.name, (255, 80, 80))
            x = obj.x * scale
            y = obj.y * scale
            r = 14
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
            draw.line((x - r - 8, y, x - r - 1, y), fill=color, width=2)
            draw.line((x + r + 1, y, x + r + 8, y), fill=color, width=2)
            label = obj.name
            tx, ty = x + r + 6, y - r - 2
            draw.rectangle((tx - 2, ty - 2, tx + 8 * len(label) + 4, ty + 15), fill=(0, 0, 0))
            draw.text((tx, ty), label, fill=color)
        gif_frames.append(image)
    if not gif_frames:
        raise ValueError("No frames available for known-object annotation GIF")
    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:], duration=650, loop=0)
    return path


def _save_inverted_jpeg(data: np.ndarray, path: Path) -> None:
    lo, hi = _shared_display_range([data])
    stretched = _stretch_with_range(data, lo, hi)
    Image.fromarray(255 - stretched, mode="L").save(path, quality=95)


def _write_blink_gif(frames: list[object], path: Path, *, max_size: int, invert: bool = False) -> None:
    images = []
    lo, hi = _shared_display_range([frame.data for frame in frames])
    for index, frame in enumerate(frames, start=1):
        stretched = _stretch_with_range(frame.data, lo, hi)
        if invert:
            stretched = 255 - stretched
        image = Image.fromarray(stretched, mode="L").convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 112, 34), fill=(0, 0, 0))
        draw.text((14, 14), f"Frame {index:02d}", fill=(255, 255, 255))
        images.append(image)
    if not images:
        raise ValueError("No frames available for GIF")
    images[0].save(path, save_all=True, append_images=images[1:], duration=450, loop=0)


def _shared_display_range(arrays: list[np.ndarray], percentile: tuple[float, float] = (0.5, 99.5)) -> tuple[float, float]:
    samples = []
    rng = np.random.default_rng(12345)
    for data in arrays:
        finite = np.asarray(data, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        if finite.size > 250_000:
            finite = rng.choice(finite, size=250_000, replace=False)
        samples.append(finite)
    if not samples:
        return 0.0, 1.0
    joined = np.concatenate(samples)
    lo, hi = np.percentile(joined, percentile)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _stretch_with_range(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(data, dtype=np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
