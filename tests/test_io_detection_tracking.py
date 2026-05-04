from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
import requests

from asteroidfinder.alignment import align_images
from asteroidfinder.calibration import build_persistent_hot_pixel_mask, calibrate_images_with_persistent_hot_pixels, remove_hot_pixels
from asteroidfinder.diagnostics import plot_track_diagnostics
from asteroidfinder.doctor import recommend_index_series, run_doctor
from asteroidfinder.detection import detect_sources
from asteroidfinder.io import load_image, save_fits
from asteroidfinder import known_objects as known_object_module
from asteroidfinder.known_objects import KnownObject, predict_known_objects_for_frames
from asteroidfinder.mpc import write_detected_track_mpc
from asteroidfinder.photometry import aperture_photometry
from asteroidfinder.tracking import track_moving_objects
from asteroidfinder_desktop.session import natural_sorted


def test_load_fits_color_cube_as_luminance(tmp_path: Path) -> None:
    path = tmp_path / "color.fit"
    data = np.stack(
        [
            np.full((8, 9), 1, dtype=np.float32),
            np.full((8, 9), 4, dtype=np.float32),
            np.full((8, 9), 7, dtype=np.float32),
        ]
    )
    fits.writeto(path, data)

    image = load_image(path)

    assert image.data.shape == (8, 9)
    assert np.allclose(image.data, 4)


def test_plate_solver_uses_2d_input_and_header_hints(tmp_path: Path, monkeypatch) -> None:
    from asteroidfinder import platesolve

    path = tmp_path / "color.fit"
    data = np.stack(
        [
            np.full((8, 9), 1, dtype=np.float32),
            np.full((8, 9), 4, dtype=np.float32),
            np.full((8, 9), 7, dtype=np.float32),
        ]
    )
    header = fits.Header()
    header["RA"] = "07 30 00.00"
    header["DEC"] = "+50 00 00.0"
    header["FOCALLEN"] = 635.0
    header["XPIXSZ"] = 3.76
    header["XBINNING"] = 1
    fits.writeto(path, data, header)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        solved_path = Path(cmd[cmd.index("--dir") + 1]) / f"{Path(cmd[-1]).stem}.new"
        solved_header = fits.Header()
        solved_header["CTYPE1"] = "RA---TAN"
        solved_header["CTYPE2"] = "DEC--TAN"
        solved_header["CRPIX1"] = 4.5
        solved_header["CRPIX2"] = 4.0
        solved_header["CRVAL1"] = 112.5
        solved_header["CRVAL2"] = 50.0
        solved_header["CDELT1"] = -0.00034
        solved_header["CDELT2"] = 0.00034
        fits.writeto(solved_path, np.zeros((8, 9), dtype=np.float32), solved_header)

    monkeypatch.setattr(platesolve.shutil, "which", lambda name: "/usr/bin/solve-field")
    monkeypatch.setattr(platesolve.subprocess, "run", fake_run)

    solution = platesolve.solve_image(path, output_dir=tmp_path / "solved", timeout=30)

    cmd = captured["cmd"]
    assert solution.wcs.has_celestial
    assert cmd[-1].endswith("-solveinput.fits")
    assert fits.getdata(cmd[-1]).shape == (8, 9)
    assert "--scale-low" in cmd
    assert "--ra" in cmd and "112.50000000" in cmd
    assert "--dec" in cmd and "50.00000000" in cmd
    assert "--downsample" in cmd and "2" in cmd
    assert "--objs" in cmd and "200" in cmd


def test_plate_solver_missing_solution_error_recommends_indexes(tmp_path: Path, monkeypatch) -> None:
    from asteroidfinder import platesolve

    path = tmp_path / "narrow.fit"
    header = fits.Header()
    header["RA"] = "21 45 00.00"
    header["DEC"] = "-15 00 00.0"
    header["FOCALLEN"] = 2262.0
    header["XPIXSZ"] = 9.0
    header["XBINNING"] = 1
    fits.writeto(path, np.zeros((2048, 3072), dtype=np.float32), header)
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "index-4210.fits").write_bytes(b"")

    def fake_run(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("--dir") + 1])
        Path(cmd[-1]).with_suffix(".axy").write_bytes(b"axy")
        return platesolve.subprocess.CompletedProcess(cmd, 0, stdout="Field 1 did not solve.", stderr="")

    monkeypatch.setattr(platesolve.shutil, "which", lambda name: "/usr/bin/solve-field")
    monkeypatch.setattr(platesolve.subprocess, "run", fake_run)

    try:
        platesolve.solve_image(path, output_dir=tmp_path / "solved", index_dir=index_dir, timeout=30)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("solve_image should fail when astrometry produces no .new file")

    assert "did not find an astrometric solution" in message
    assert "narrow-solveinput.axy" in message
    assert "recommended 4200-series indexes" in message
    assert "4206, 4207, 4208" in message
    assert "Installed/visible index series: 4210" in message


def test_detect_sources_finds_synthetic_stars() -> None:
    image = np.zeros((100, 100), dtype=np.float32)
    yy, xx = np.indices(image.shape)
    for x, y in [(25, 30), (70, 60)]:
        image += 500 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4)
    image += np.random.default_rng(123).normal(10, 1, image.shape).astype(np.float32)

    sources = detect_sources(image, sigma=5)

    assert len(sources) >= 2
    coords = {(round(src.x), round(src.y)) for src in sources[:2]}
    assert (25, 30) in coords
    assert (70, 60) in coords


def test_hot_pixel_cleanup_replaces_isolated_spike() -> None:
    image = np.full((21, 21), 100, dtype=np.float32)
    image[10, 10] = 5000

    cleaned, mask = remove_hot_pixels(image, sigma=5)

    assert mask[10, 10]
    assert cleaned[10, 10] == 100


def test_persistent_hot_pixel_mask_detects_dead_pixel(tmp_path: Path) -> None:
    paths = []
    for frame in range(4):
        image = np.full((21, 21), 1000, dtype=np.float32)
        image[7, 11] = 0
        path = tmp_path / f"dead_{frame}.fits"
        save_fits(image, path)
        paths.append(path)

    mask = build_persistent_hot_pixel_mask(paths, sigma=8, neighbor_sigma=4, min_center_neighbor_ratio=2, min_frames=3)
    results = calibrate_images_with_persistent_hot_pixels(paths, hot_sigma=8, neighbor_sigma=4, min_center_neighbor_ratio=2, min_frames=3)

    assert mask[7, 11]
    assert all(result.data[7, 11] == 1000 for result in results)


def test_persistent_hot_pixel_mask_does_not_flag_moving_star(tmp_path: Path) -> None:
    paths = []
    yy, xx = np.indices((50, 50))
    for frame in range(4):
        image = np.full((50, 50), 100, dtype=np.float32)
        image[8, 9] = 8000
        star_x = 20 + frame
        star_y = 25
        image += 3000 * np.exp(-((xx - star_x) ** 2 + (yy - star_y) ** 2) / 2)
        path = tmp_path / f"frame_{frame}.fits"
        save_fits(image, path)
        paths.append(path)

    mask = build_persistent_hot_pixel_mask(paths, sigma=8, neighbor_sigma=4, min_center_neighbor_ratio=2, min_frames=3)
    results = calibrate_images_with_persistent_hot_pixels(paths, hot_sigma=8, neighbor_sigma=4, min_center_neighbor_ratio=2, min_frames=3)

    assert mask[8, 9]
    assert not mask[25, 20]
    assert sum(int(result.hot_pixel_mask.sum()) for result in results) == 4


def test_persistent_hot_pixel_calibration_writes_qa_masks(tmp_path: Path) -> None:
    paths = []
    for frame in range(3):
        image = np.full((20, 20), 100, dtype=np.float32)
        image[5, 6] = 8000
        image[10, 8 + frame] = 7000
        path = tmp_path / f"qa_{frame}.fits"
        save_fits(image, path)
        paths.append(path)

    out_dir = tmp_path / "calibrated"
    calibrate_images_with_persistent_hot_pixels(
        paths,
        output_dir=out_dir,
        hot_sigma=8,
        neighbor_sigma=4,
        min_center_neighbor_ratio=2,
        min_frames=2,
    )

    qa_dir = out_dir / "hot_pixel_qa"
    summary = qa_dir / "hot_pixel_summary.csv"
    assert summary.exists()
    text = summary.read_text()
    assert "persistent_mask_total" in text
    assert "transient_hits_in_frame" in text
    assert (qa_dir / "persistent_mask_20x20.png").exists()
    assert (qa_dir / "transient_union_20x20.png").exists()
    assert (qa_dir / "qa_0_hot_pixel_classified.png").exists()
    from PIL import Image

    persistent_preview = Image.open(qa_dir / "persistent_mask_20x20.png").convert("L")
    classified_preview = Image.open(qa_dir / "qa_0_hot_pixel_classified.png").convert("RGB")
    assert persistent_preview.getpixel((0, 0)) == 255
    assert persistent_preview.getpixel((6, 5)) == 0
    assert classified_preview.getpixel((6, 5)) == (0, 0, 0)
    assert classified_preview.getpixel((8, 10)) == (150, 150, 150)


def test_persistent_hot_pixel_calibration_handles_mixed_shapes(tmp_path: Path) -> None:
    paths = []
    for index, shape in enumerate([(20, 20), (20, 20), (12, 16)]):
        image = np.full(shape, 100, dtype=np.float32)
        image[3, 4] = 8000
        path = tmp_path / f"mixed_{index}.fits"
        save_fits(image, path)
        paths.append(path)

    results = calibrate_images_with_persistent_hot_pixels(paths, hot_sigma=8, neighbor_sigma=4, min_center_neighbor_ratio=2)

    assert [result.data.shape for result in results] == [(20, 20), (20, 20), (12, 16)]


def test_hot_pixel_mask_replacement_ignores_masked_neighbors() -> None:
    from asteroidfinder.calibration import apply_hot_pixel_mask

    image = np.full((7, 7), 100, dtype=np.float32)
    image[3, 3] = 9000
    image[3, 4] = 8000
    mask = np.zeros(image.shape, dtype=bool)
    mask[3, 3] = True
    mask[3, 4] = True

    cleaned = apply_hot_pixel_mask(image, mask)

    assert cleaned[3, 3] == 100
    assert cleaned[3, 4] == 100


def test_aperture_photometry_measures_positive_flux() -> None:
    image = np.full((80, 80), 20, dtype=np.float32)
    yy, xx = np.indices(image.shape)
    image += 500 * np.exp(-((xx - 40) ** 2 + (yy - 35) ** 2) / 4)
    source = detect_sources(image, sigma=5)[0]

    phot = aperture_photometry(image, source)

    assert phot.net_flux > 0
    assert phot.snr > 10


def test_track_moving_object_on_synthetic_sequence(tmp_path: Path) -> None:
    paths = []
    yy, xx = np.indices((120, 120))
    stars = [(20, 20), (80, 30), (45, 90), (95, 95), (15, 75)]
    for frame in range(4):
        image = np.zeros((120, 120), dtype=np.float32) + 20
        for x, y in stars:
            image += 700 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 5)
        moving_x = 30 + frame * 5
        moving_y = 40 + frame * 3
        image += 600 * np.exp(-((xx - moving_x) ** 2 + (yy - moving_y) ** 2) / 5)
        path = tmp_path / f"frame_{frame}.fits"
        save_fits(image, path)
        paths.append(path)

    tracks = track_moving_objects(paths, sigma=5, min_detections=3)

    assert tracks
    best = tracks[0]
    assert len(best.detections) >= 3
    assert abs(best.velocity_x - 5) < 1.5
    assert abs(best.velocity_y - 3) < 1.5
    assert best.detections[0].photometry is not None


def test_track_moving_object_can_skip_alignment_for_aligned_sequence(tmp_path: Path) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)

    tracks = track_moving_objects(paths, sigma=5, min_detections=3, assume_aligned=True)

    assert tracks
    assert tracks[0].detections[0].photometry is not None


def test_align_images_uses_wcs_reprojection_when_available(tmp_path: Path) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)

    aligned = align_images(paths, prefer_translation=False)

    assert aligned[1].method == "wcs-reproject"
    assert aligned[1].footprint is not None
    assert aligned[1].data.shape == aligned[0].data.shape


def test_align_images_writes_alignment_qa(tmp_path: Path) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)

    align_images(paths, output_dir=tmp_path / "aligned", prefer_translation=False)

    qa_path = tmp_path / "aligned" / "alignment_qa.csv"
    assert qa_path.exists()
    text = qa_path.read_text()
    assert "rms_error_px" in text
    assert "wcs-reproject" in text


def test_align_images_ignores_isolated_hot_pixels_in_star_matching(tmp_path: Path) -> None:
    paths = []
    yy, xx = np.indices((160, 160))
    stars = [(30, 30), (120, 35), (45, 120), (125, 125), (80, 70), (25, 95)]
    shift_x, shift_y = 4, -3
    hot_pixels = [(8, 8), (150, 20), (16, 140), (141, 142), (80, 150), (100, 12)]
    for frame in range(2):
        image = np.zeros((160, 160), dtype=np.float32) + 20
        for x, y in stars:
            sx = x + (shift_x if frame else 0)
            sy = y + (shift_y if frame else 0)
            image += 800 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 5)
        for index, (x, y) in enumerate(hot_pixels):
            image[y, x] = 15000 + index * 100
        path = tmp_path / f"frame_{frame + 1}.fits"
        save_fits(image, path)
        paths.append(path)

    aligned = align_images(paths, prefer_translation=False)

    assert aligned[1].method == "astroalign"
    assert aligned[1].rms_error is not None
    assert aligned[1].rms_error < 1.0


def test_natural_sorted_keeps_numeric_frame_order() -> None:
    paths = [Path("frame_10.fits"), Path("frame_2.fits"), Path("frame_1.fits")]

    assert [path.name for path in natural_sorted(paths)] == ["frame_1.fits", "frame_2.fits", "frame_10.fits"]


def test_detected_track_mpc_export_uses_measured_track(tmp_path: Path) -> None:
    paths = []
    yy, xx = np.indices((120, 120))
    stars = [(20, 20), (80, 30), (45, 90), (95, 95), (15, 75)]
    for frame in range(3):
        image = np.zeros((120, 120), dtype=np.float32) + 20
        for x, y in stars:
            image += 700 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 5)
        image += 600 * np.exp(-((xx - (30 + frame * 5)) ** 2 + (yy - (40 + frame * 3)) ** 2) / 5)
        header = fits.Header()
        header["DATE-OBS"] = f"2026-01-01T00:0{frame}:00"
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 60.0
        header["CRPIX2"] = 60.0
        header["CRVAL1"] = 100.0
        header["CRVAL2"] = 20.0
        header["CDELT1"] = -0.00028
        header["CDELT2"] = 0.00028
        header["FILTER"] = "r"
        header["MAGZP"] = 25.0
        path = tmp_path / f"frame_{frame}.fits"
        save_fits(image, path, header)
        paths.append(path)

    aligned = align_images(paths)
    tracks = track_moving_objects(paths, sigma=5, min_detections=3)
    mpc_path = tmp_path / "measured_mpc.txt"
    csv_path = tmp_path / "measured.csv"

    write_detected_track_mpc(tracks, aligned, mpc_path, observatory_code="500", csv_path=csv_path)

    text = mpc_path.read_text()
    assert "measured centroids" in text
    assert "AF0001" in text
    rows = csv_path.read_text().splitlines()
    assert rows[0].startswith("track_id,frame_index")
    assert len(rows) >= 4
    assert ",calibrated_magzp," in rows[1]


def test_alignment_outputs_preserve_per_frame_observation_time(tmp_path: Path) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)
    out_dir = tmp_path / "aligned"

    align_images(paths, output_dir=out_dir)

    aligned_paths = [out_dir / f"{path.stem}_aligned.fits" for path in paths]
    original_dates = [fits.getheader(path)["DATE-OBS"] for path in paths]
    aligned_dates = [fits.getheader(path)["DATE-OBS"] for path in aligned_paths]
    assert aligned_dates == original_dates
    assert fits.getheader(aligned_paths[1])["CTYPE1"] == "RA---TAN"

    anchor = KnownObject(
        frame=aligned_paths[0],
        date_obs="2026-01-01T00:00:00.000",
        number="1",
        name="TestAst",
        object_type="asteroid",
        ra_deg=100.0,
        dec_deg=20.0,
        x=60.0,
        y=60.0,
        v_mag=18.0,
        center_distance_arcsec=None,
        ra_rate_arcsec_per_hour=36.0,
        dec_rate_arcsec_per_hour=0.0,
    )
    predicted = predict_known_objects_for_frames([anchor], aligned_paths)
    assert predicted[2].x != predicted[0].x


def test_track_diagnostics_writes_png_and_summary(tmp_path: Path) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)
    tracks = track_moving_objects(paths, sigma=5, min_detections=3)

    written = plot_track_diagnostics(tracks, tmp_path / "diag")

    assert written
    assert written[0].exists()
    summary = (tmp_path / "diag" / "track_diagnostics.csv").read_text()
    assert "pixel_speed" in summary
    assert "sky_speed" in summary


def test_doctor_reports_index_recommendation(tmp_path: Path) -> None:
    sample = tmp_path / "sample.fits"
    save_fits(np.zeros((100, 200), dtype=np.float32), sample)
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "index-4210.fits").write_bytes(b"placeholder")

    checks = run_doctor(index_dir=index_dir, sample_image=sample, scale_low=1.0, scale_high=1.5)
    series = recommend_index_series(image_width_px=6248, scale_low=1.0, scale_high=1.5)

    assert any(check.name == "astrometry index files" and check.ok for check in checks)
    assert series == ["4210", "4211", "4212"]


def test_doctor_recommends_split_indexes_for_t30_scale() -> None:
    series = recommend_index_series(image_width_px=3072, scale_low=0.701, scale_high=0.948)

    assert series == ["4206", "4207", "4208"]


def test_known_object_time_fallback_accepts_obsjd() -> None:
    from astropy.io import fits
    from asteroidfinder.known_objects import _observation_time

    header = fits.Header()
    header["OBSJD"] = 2460675.5

    assert _observation_time(header).isot.startswith("2024-")


def test_known_object_prediction_uses_rates_without_requery(tmp_path: Path) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)
    anchor = KnownObject(
        frame=paths[0],
        date_obs="2026-01-01T00:00:00.000",
        number="1",
        name="TestAst",
        object_type="asteroid",
        ra_deg=100.0,
        dec_deg=20.0,
        x=60.0,
        y=60.0,
        v_mag=18.0,
        center_distance_arcsec=None,
        ra_rate_arcsec_per_hour=36.0,
        dec_rate_arcsec_per_hour=0.0,
    )

    predicted = predict_known_objects_for_frames([anchor], paths)

    assert len(predicted) == 3
    assert [obj.frame for obj in predicted] == paths
    assert predicted[2].x != predicted[0].x


def test_known_object_query_retries_skybot_http_500(tmp_path: Path, monkeypatch) -> None:
    from astroquery.imcce import Skybot

    path = _synthetic_wcs_sequence(tmp_path)[0]
    calls = []

    def fake_cone_search(*args, **kwargs):
        calls.append(kwargs.get("cache"))
        if len(calls) == 1:
            response = requests.Response()
            response.status_code = 500
            raise requests.exceptions.HTTPError("500 Server Error", response=response)
        return Table(
            {
                "Number": ["123"],
                "Name": ["RetryAst"],
                "Type": ["MB"],
                "RA": [100.0],
                "DEC": [20.0],
                "V": [18.0],
                "centerdist": [0.0],
                "RA_rate": [36.0],
                "DEC_rate": [0.0],
            }
        )

    monkeypatch.setattr(Skybot, "cone_search", fake_cone_search)
    monkeypatch.setattr(known_object_module.time, "sleep", lambda *_: None)

    objects = known_object_module.query_known_objects_in_frame(path)

    assert len(objects) == 1
    assert objects[0].name == "RetryAst"
    assert calls == [True, False]


def test_known_object_motion_cache_tries_alternate_anchor_after_server_error(tmp_path: Path, monkeypatch) -> None:
    paths = _synthetic_wcs_sequence(tmp_path)
    calls = []

    def fake_query(path: Path, **kwargs):
        calls.append(Path(path))
        if len(calls) == 1:
            raise known_object_module.KnownObjectQueryError("SkyBoT server error")
        return [
            KnownObject(
                frame=Path(path),
                date_obs="2026-01-01T00:00:00.000",
                number="1",
                name="FallbackAst",
                object_type="asteroid",
                ra_deg=100.0,
                dec_deg=20.0,
                x=60.0,
                y=60.0,
                v_mag=18.0,
                center_distance_arcsec=None,
                ra_rate_arcsec_per_hour=36.0,
                dec_rate_arcsec_per_hour=0.0,
            )
        ]

    monkeypatch.setattr(known_object_module, "query_known_objects_in_frame", fake_query)

    predicted = known_object_module.query_known_objects_with_motion_cache(paths, anchor_index=1)

    assert calls[:2] == [paths[1], paths[0]]
    assert len(predicted) == 3
    assert {obj.name for obj in predicted} == {"FallbackAst"}


def _synthetic_wcs_sequence(tmp_path: Path) -> list[Path]:
    paths = []
    yy, xx = np.indices((120, 120))
    stars = [(20, 20), (80, 30), (45, 90), (95, 95), (15, 75)]
    for frame in range(3):
        image = np.zeros((120, 120), dtype=np.float32) + 20
        for x, y in stars:
            image += 700 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 5)
        image += 600 * np.exp(-((xx - (30 + frame * 5)) ** 2 + (yy - (40 + frame * 3)) ** 2) / 5)
        header = fits.Header()
        header["DATE-OBS"] = f"2026-01-01T00:0{frame}:00"
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 60.0
        header["CRPIX2"] = 60.0
        header["CRVAL1"] = 100.0
        header["CRVAL2"] = 20.0
        header["CDELT1"] = -0.00028
        header["CDELT2"] = 0.00028
        header["FILTER"] = "r"
        header["MAGZP"] = 25.0
        path = tmp_path / f"wcs_frame_{frame}.fits"
        save_fits(image, path, header)
        paths.append(path)
    return paths
