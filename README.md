# AsteroidFinder

AsteroidFinder is a Python library, CLI, and desktop app for reducing asteroid
image sequences. It helps you clean FITS frames, plate solve them, align stars,
detect moving objects, check whether detections match known solar-system
objects, and prepare measured-track outputs for later MPC review.

It does real processing. Plate solving uses embedded FITS WCS when available or
local astrometry.net `solve-field` when configured. If a frame cannot be solved,
AsteroidFinder reports that instead of pretending.

## Workflow

1. Import a matching set of FITS frames.
2. Clean calibration artifacts such as persistent hot pixels.
3. Plate solve frames so pixel positions can be tied to sky coordinates.
4. Align stars so real movers stand out between frames.
5. Detect moving-object tracks.
6. Query known objects and mark detections as known or unknown.
7. Inspect movement charts, measurements, and residuals.
8. Export measured observations for MPC-style review.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

For the desktop app:

```bash
python3 -m pip install -e ".[desktop]"
asteroidfinder-desktop
```

## Desktop App

The desktop app is the easiest way to use the project. It focuses on the main
reduction path:

- clean hot pixels
- plate solve
- align stars
- detect movers
- identify known matches
- export MPC-style measured tracks
- open reports and movement charts

![AsteroidFinder desktop main view](docs/main.png)

![AsteroidFinder desktop inverted view](docs/main-inverted.png)

## CLI Examples

Inspect frames:

```bash
asteroidfinder inspect data/raw/*.fit
```

Solve one frame:

```bash
asteroidfinder solve data/raw/example.fit --out solved
```

Align a sequence:

```bash
asteroidfinder align data/raw/*.fit --out aligned
```

Detect moving objects:

```bash
asteroidfinder track data/raw/*.fit --out tracks.csv
```

Run the broader asteroid workflow:

```bash
asteroidfinder asteroid-run data/raw/*.fit --out asteroidfinder_output
```

## Plate Solving

For blind solving, install astrometry.net and download index files that match
your field of view. `solve-field` must be on your `PATH`.

Check your setup:

```bash
asteroidfinder doctor \
  --index-dir ~/astrometry-indexes/4200 \
  --sample-image data/raw/example.fit \
  --scale-low 1.0 \
  --scale-high 1.5
```

Download a starter 4200-series index file:

```bash
asteroidfinder install-indexes --index-dir ~/astrometry-indexes/4200 --series 4210
```

Then solve with the index directory:

```bash
asteroidfinder solve data/raw/example.fit \
  --index-dir ~/astrometry-indexes/4200 \
  --scale-low 1.0 \
  --scale-high 1.5
```

## Outputs

Typical output folders contain:

- cleaned/calibrated FITS frames
- solved FITS files with WCS
- aligned FITS frames
- moving-object track CSVs
- known-object match CSVs
- MPC-style measured-track exports
- movement diagnostic charts
- an HTML/report window summary

## Python API

```python
from asteroidfinder import (
    align_images,
    load_image,
    plot_track_diagnostics,
    solve_image,
    track_moving_objects,
)

frame = load_image("data/raw/image.fit")
solution = solve_image("data/raw/image.fit")
aligned = align_images(["data/raw/001.fit", "data/raw/002.fit"])
tracks = track_moving_objects([
    "data/raw/001.fit",
    "data/raw/002.fit",
    "data/raw/003.fit",
])
plot_track_diagnostics(tracks, "diagnostics")
```

## Demos

- `demo/` runs the local telescope workflow.
- `demo2/` runs the clean ZTF Photographica workflow.

Demo outputs are useful for checking whether solving, alignment, known-object
matching, and mover detection are behaving as expected on real data.
