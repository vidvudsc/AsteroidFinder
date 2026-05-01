# AsteroidFinder Desktop UI Plan

## Goal

Build a fast local desktop app on top of the existing `asteroidfinder` library. The desktop app should feel closer to Tycho or ASTAP than a notebook: open a night of FITS frames, inspect them, solve, align, blink, detect moving objects, compare known objects, and export measured observations.

The GUI should call the library. It should not duplicate the science code.

## Proposed Stack

- UI toolkit: PySide6.
- Image viewer: `QGraphicsView` / `QGraphicsScene` with cached stretched `QImage` or `QPixmap` layers.
- Work scheduling: `QThreadPool` with `QRunnable` workers for solve, align, calibration, tracking, report generation, and exports.
- Project state: a small session model saved as JSON next to the output folder.
- Core processing: existing `asteroidfinder` package functions and CLI workflow code.

PySide6 is an optional dependency because users should still be able to install the command-line library without Qt.

## First MVP

1. Open a folder of FITS files.
2. Show the frame list with metadata: time, filter, WCS status, size, exposure, solved or unsolved.
3. Display a stretched FITS preview with invert, percentile stretch, zoom, pan, and frame slider.
4. Blink frames with play, pause, speed, and previous/next controls.
5. Run calibration with persistent hot-pixel masking.
6. Run plate solving with index-dir settings and visible progress logs.
7. Align frames and display no-data borders clearly instead of silently cropping.
8. Run moving-object tracking on aligned frames.
9. Show detected tracks in a table with speed, angle, detections, RA/Dec fit, and quality flags.
10. Overlay detected tracks, known-object predictions, forced-photometry apertures, and selected-track labels.
11. Export measured-track MPC text and CSV.
12. Generate and open the HTML report.

## Main Window Layout

- Left workflow panel: input folder, output folder, solve settings, calibration settings, tracking settings, and run buttons.
- Center image viewer: FITS preview, blink mode, overlays, zoom, pan, and pixel/sky coordinate readout.
- Right inspection panel: selected frame metadata, selected track details, known-object matches, forced photometry, and diagnostic plots.
- Bottom log panel: command progress, warnings, solver output summary, and clickable output paths.

## Session Model

The app should keep a `SessionState` object with:

- `input_dir`
- `output_dir`
- `frames`
- `calibrated_frames`
- `solved_frames`
- `aligned_frames`
- `known_objects`
- `forced_photometry`
- `detected_tracks`
- `selected_frame_index`
- `selected_track_id`
- pipeline settings

Each long-running worker should return a typed result object instead of mutating the UI directly.

## Viewer Behavior

- Use percentile stretch by default.
- Cache one display image per frame and stretch setting.
- Keep overlay geometry separate from image pixels.
- Reproject RA/Dec overlays through WCS at draw time.
- Render track paths with frame-index markers.
- Render selected tracks with a stronger outline and matching table selection.
- Preserve no-data borders after alignment so users can see what data is real.

## First Screens To Build

1. `desktop/main.py`: app entry point.
2. `desktop/session.py`: session model and JSON save/load.
3. `desktop/workers.py`: background worker wrapper.
4. `desktop/viewer.py`: FITS image viewer and overlay drawing.
5. `desktop/main_window.py`: workflow layout and actions.

## Milestones

### Milestone 1: Viewer

- Open FITS folder.
- Display image previews.
- Blink frames smoothly.
- Toggle invert and stretch.

### Milestone 2: Pipeline Controls

- Run solve, align, calibrate, and track from the UI.
- Show progress and failures without freezing.
- Reuse the existing output folder layout.

### Milestone 3: Track Analysis

- Track table.
- Per-track diagnostic plot preview.
- SkyBoT known-object comparison.
- Forced known-object overlay GIF generation.

### Milestone 4: Reporting And Export

- One-click HTML report.
- Measured-track MPC export.
- CSV diagnostics export.
- Open output folder/report from the app.

## Non-Goals For The First Version

- Automatic MPC submission.
- Full photometry reduction pipeline for publication-grade calibrated magnitudes.
- Deep image editor tools.
- Replacing the CLI.

The first GUI should make the real pipeline easier to inspect and run. The science path remains the library and the same generated output files.
