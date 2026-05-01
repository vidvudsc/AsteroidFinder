# AsteroidFinder Demo

This demo runs the local six-frame asteroid sequence through the practical
pipeline:

1. Select the six 2-D `calibrated-T68...fit` luminance frames.
2. Build a conservative persistent hot-pixel map across the sequence, then
   remove only those fixed isolated sensor defects.
3. Plate solve each frame.
   - By default, it uses embedded WCS if present.
   - Add `--force-astrometry` to force local astrometry.net `solve-field`.
4. Query IMCCE SkyBoT for known asteroids / solar-system objects expected in
   the solved field at `DATE-OBS`.
5. Align frames on the stars and crop to the common valid overlap, so aligned
   outputs do not include black/no-data borders.
6. Write aligned FITS files and JPEG previews.
7. Create `blink_aligned.gif`.
8. Write inverted previews and `blink_aligned_inverted.gif`.
9. Write black/white hot-pixel mask PNGs.
10. Write median stack and difference frames.
11. Track moving-object candidates and write `tracks.csv`.
12. Write `report.html`.

Run:

```bash
python3 demo/run_demo.py
```

Force astrometry.net solving:

```bash
python3 demo/run_demo.py --force-astrometry
```

Use a custom index directory:

```bash
python3 demo/run_demo.py --force-astrometry --index-dir /path/to/astrometry/indexes
```

If this fails with "You must list at least one index", Homebrew installed the
solver but not the astrometry.net index files. Install index files covering
about 1.0-1.5 arcsec/pixel for these frames, then rerun the command.

Use the raw unsolved frames instead of the calibrated luminance frames:

```bash
python3 demo/run_demo.py --use-raw --force-astrometry
```

Outputs go to `demo/output/`:

- `01_hot_cleaned/`
- `02_solved/`
- `03_aligned/`
- `04_previews/`
- `05_difference/`
- `06_hot_pixel_masks/`
- `07_inverted/`
- `alignment_report.csv`
- `hot_pixel_report.csv`
- `hot_pixel_coordinates.csv`
- `known_objects.csv`
- `known_objects_summary.csv`
- `known_object_forced_photometry.csv`
- `mpc_observations.txt`
- `report.html`
- `blink_aligned.gif`
- `blink_aligned_inverted.gif`
- `stack_median.fits`
- `tracks.csv`
