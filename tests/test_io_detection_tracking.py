from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from asteroidfinder.calibration import remove_hot_pixels
from asteroidfinder.detection import detect_sources
from asteroidfinder.io import load_image, save_fits
from asteroidfinder.photometry import aperture_photometry
from asteroidfinder.tracking import track_moving_objects


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
