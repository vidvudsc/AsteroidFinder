from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import align_images, stack_images
from .calibration import calibrate_images, make_master_frame
from .detection import detect_sources
from .doctor import install_astrometry_indexes, recommend_index_series, run_doctor
from .io import load_image, save_fits, save_jpeg
from .photometry import aperture_photometry
from .platesolve import solve_image
from .tracking import track_moving_objects
from .workflow import run_asteroid_workflow, write_alignment_report, write_tracks_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asteroidfinder")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor", help="Check plate solving, index files, and sample FITS readiness")
    doctor_p.add_argument("--index-dir", type=Path)
    doctor_p.add_argument("--sample-image", type=Path)
    doctor_p.add_argument("--scale-low", type=float)
    doctor_p.add_argument("--scale-high", type=float)

    index_p = sub.add_parser("install-indexes", help="Download astrometry.net 4200-series index files")
    index_p.add_argument("--index-dir", type=Path, default=Path.home() / "astrometry-indexes" / "4200")
    index_p.add_argument("--series", nargs="+", default=["4210"], help="Index series numbers, e.g. 4210 4211 4212")

    inspect_p = sub.add_parser("inspect", help="Print basic image metadata")
    inspect_p.add_argument("paths", nargs="+")

    detect_p = sub.add_parser("detect", help="Detect sources in images")
    detect_p.add_argument("paths", nargs="+")
    detect_p.add_argument("--sigma", type=float, default=3.0)

    phot_p = sub.add_parser("photometry", help="Detect sources and measure aperture photometry")
    phot_p.add_argument("path")
    phot_p.add_argument("--sigma", type=float, default=4.0)
    phot_p.add_argument("--limit", type=int, default=50)

    master_p = sub.add_parser("master", help="Build a master calibration frame")
    master_p.add_argument("paths", nargs="+")
    master_p.add_argument("--out", required=True, type=Path)
    master_p.add_argument("--method", choices=["median", "mean"], default="median")

    cal_p = sub.add_parser("calibrate", help="Apply bias/dark/flat correction and hot-pixel cleanup")
    cal_p.add_argument("paths", nargs="+")
    cal_p.add_argument("--out", type=Path, default=Path("calibrated"))
    cal_p.add_argument("--bias", type=Path)
    cal_p.add_argument("--dark", type=Path)
    cal_p.add_argument("--flat", type=Path)
    cal_p.add_argument("--hot-sigma", type=float, default=8.0)

    solve_p = sub.add_parser("solve", help="Plate solve images with embedded WCS or astrometry.net")
    solve_p.add_argument("paths", nargs="+")
    solve_p.add_argument("--out", type=Path, default=Path("solved"))
    solve_p.add_argument("--scale-low", type=float)
    solve_p.add_argument("--scale-high", type=float)
    solve_p.add_argument("--scale-units", default="arcsecperpix")
    solve_p.add_argument("--index-dir", type=Path)
    solve_p.add_argument("--timeout", type=int, default=180)

    align_p = sub.add_parser("align", help="Align images and optionally stack them")
    align_p.add_argument("paths", nargs="+")
    align_p.add_argument("--out", type=Path, default=Path("aligned"))
    align_p.add_argument("--stack", choices=["median", "mean", "sum"])
    align_p.add_argument("--preview", action="store_true")
    align_p.add_argument("--report", type=Path)

    track_p = sub.add_parser("track", help="Track moving-object candidates")
    track_p.add_argument("paths", nargs="+")
    track_p.add_argument("--out", type=Path, default=Path("tracks.csv"))
    track_p.add_argument("--sigma", type=float, default=4.0)
    track_p.add_argument("--min-detections", type=int, default=3)

    run_p = sub.add_parser("asteroid-run", help="Calibrate, align, stack, difference, and track asteroids")
    run_p.add_argument("paths", nargs="+")
    run_p.add_argument("--out", type=Path, default=Path("asteroid_run"))
    run_p.add_argument("--bias", type=Path)
    run_p.add_argument("--dark", type=Path)
    run_p.add_argument("--flat", type=Path)
    run_p.add_argument("--hot-sigma", type=float, default=8.0)
    run_p.add_argument("--sigma", type=float, default=4.0)
    run_p.add_argument("--min-detections", type=int, default=3)
    run_p.add_argument("--no-preview", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "install-indexes":
        return _install_indexes(args)
    if args.command == "inspect":
        return _inspect(args.paths)
    if args.command == "detect":
        return _detect(args.paths, sigma=args.sigma)
    if args.command == "photometry":
        return _photometry(args.path, sigma=args.sigma, limit=args.limit)
    if args.command == "master":
        return _master(args.paths, args)
    if args.command == "calibrate":
        return _calibrate(args.paths, args)
    if args.command == "solve":
        return _solve(args.paths, args)
    if args.command == "align":
        return _align(args.paths, args)
    if args.command == "track":
        return _track(args.paths, args)
    if args.command == "asteroid-run":
        return _asteroid_run(args.paths, args)
    raise AssertionError(args.command)


def _doctor(args: argparse.Namespace) -> int:
    checks = run_doctor(
        index_dir=args.index_dir,
        sample_image=args.sample_image,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    ok = True
    for check in checks:
        mark = "ok" if check.ok else "fail"
        print(f"{mark:4} {check.name}: {check.detail}")
        ok = ok and check.ok
    if args.sample_image and args.scale_low and args.scale_high:
        image = load_image(args.sample_image)
        series = recommend_index_series(
            image_width_px=max(image.data.shape),
            scale_low=args.scale_low,
            scale_high=args.scale_high,
        )
        print(f"recommended 4200 index series: {' '.join(series)}")
    return 0 if ok else 1


def _install_indexes(args: argparse.Namespace) -> int:
    paths = install_astrometry_indexes(args.series, args.index_dir)
    print(f"index_dir={args.index_dir.expanduser()}")
    for path in paths:
        print(f"index={path}")
    return 0


def _inspect(paths: list[str]) -> int:
    for path in paths:
        image = load_image(path)
        print(f"{image.path}: shape={image.data.shape} dtype={image.data.dtype}")
        if image.header is not None:
            for key in ("DATE-OBS", "EXPTIME", "RA", "DEC", "OBJCTRA", "OBJCTDEC", "FOCALLEN", "XPIXSZ"):
                if key in image.header:
                    print(f"  {key}={image.header[key]}")
    return 0


def _detect(paths: list[str], *, sigma: float) -> int:
    for path in paths:
        image = load_image(path)
        sources = detect_sources(image.data, sigma=sigma)
        print(f"{image.path}: {len(sources)} sources")
        for source in sources[:20]:
            print(f"  x={source.x:.2f} y={source.y:.2f} flux={source.flux:.1f} snr={source.snr:.1f}")
    return 0


def _photometry(path: str, *, sigma: float, limit: int) -> int:
    image = load_image(path)
    sources = detect_sources(image.data, sigma=sigma, max_sources=limit)
    print("index,x,y,flux,net_flux,snr,instrumental_mag")
    for index, source in enumerate(sources, start=1):
        phot = aperture_photometry(image.data, source)
        mag = "" if phot.instrumental_mag is None else f"{phot.instrumental_mag:.5f}"
        print(
            f"{index},{source.x:.3f},{source.y:.3f},{source.flux:.3f},"
            f"{phot.net_flux:.3f},{phot.snr:.3f},{mag}"
        )
    return 0


def _master(paths: list[str], args: argparse.Namespace) -> int:
    master = make_master_frame(paths, method=args.method)
    save_fits(master, args.out)
    print(f"wrote master frame to {args.out}")
    return 0


def _calibrate(paths: list[str], args: argparse.Namespace) -> int:
    results = calibrate_images(
        paths,
        output_dir=args.out,
        master_bias=args.bias,
        master_dark=args.dark,
        master_flat=args.flat,
        hot_sigma=args.hot_sigma,
    )
    hot_pixels = sum(int(result.hot_pixel_mask.sum()) for result in results)
    print(f"calibrated {len(results)} images into {args.out}; replaced {hot_pixels} hot pixels")
    return 0


def _solve(paths: list[str], args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        solution = solve_image(
            path,
            output_dir=args.out,
            index_dir=args.index_dir,
            timeout=args.timeout,
            scale_low=args.scale_low,
            scale_high=args.scale_high,
            scale_units=args.scale_units,
        )
        center = solution.wcs.pixel_to_world_values(0, 0)
        print(f"{solution.path}: solved by {solution.method}; origin ra={center[0]:.6f} dec={center[1]:.6f}")
        if solution.solved_fits is not None:
            print(f"  solved_fits={solution.solved_fits}")
    return 0


def _align(paths: list[str], args: argparse.Namespace) -> int:
    frames = align_images(paths, output_dir=args.out)
    print(f"aligned {len(frames)} images into {args.out}")
    report = args.report or (args.out / "alignment_report.csv")
    write_alignment_report(frames, report)
    print(f"alignment_report={report}")
    if args.stack:
        stacked = stack_images(frames, method=args.stack)
        fits_path = args.out / f"stack_{args.stack}.fits"
        save_fits(stacked, fits_path)
        print(f"stack={fits_path}")
        if args.preview:
            preview_path = args.out / f"stack_{args.stack}.jpg"
            save_jpeg(stacked, preview_path)
            print(f"preview={preview_path}")
    return 0


def _track(paths: list[str], args: argparse.Namespace) -> int:
    tracks = track_moving_objects(paths, sigma=args.sigma, min_detections=args.min_detections)
    write_tracks_csv(tracks, args.out)
    print(f"wrote {len(tracks)} tracks to {args.out}")
    return 0


def _asteroid_run(paths: list[str], args: argparse.Namespace) -> int:
    result = run_asteroid_workflow(
        paths,
        output_dir=args.out,
        master_bias=args.bias,
        master_dark=args.dark,
        master_flat=args.flat,
        hot_sigma=args.hot_sigma,
        sigma=args.sigma,
        min_detections=args.min_detections,
        make_preview=not args.no_preview,
    )
    print(f"aligned={len(result.aligned_frames)}")
    print(f"tracks={len(result.tracks)}")
    print(f"stack={result.stack_path}")
    print(f"tracks_csv={result.tracks_path}")
    print(f"alignment_report={result.alignment_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
