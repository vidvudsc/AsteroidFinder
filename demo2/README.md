# Demo 2: ZTF Photographica Test

This demo uses the exact ZTF downloader command requested:

```bash
python3 ztf.py \
  --ra 232.89007833333332 \
  --dec=-16.83565 \
  --date 2024-12-30 \
  --filter zr \
  --cutout-size 12arcmin \
  --out-dir data/Photographica/input
```

The command downloads three real ZTF `zr` FITS cutouts around asteroid
`Photographica (443)`. The frames include embedded ZTF WCS, so the demo validates
plate solving through embedded WCS, star alignment, moving-object tracking, and
SkyBoT known-object comparison.

Run:

```bash
python demo2/run_demo2.py
```

Outputs are written to `demo2/output/`.
